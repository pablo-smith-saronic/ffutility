#![cfg(all(target_os = "linux", feature = "v4l"))]

use anyhow::{bail, Result};

use bytes::{Bytes, BytesMut};

use crate::encoders::{EncoderConfig, EncoderType, FfmpegOptions, VideoEncoder, InputType};

use ffmpeg_next::util::format::Pixel as AvPixel;

use tracing::{debug, error};

use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::traits::Capture;

use std::fs::File;
use std::{io, io::Write};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::io::StreamReader;

// TODO: make this more generic so you can have a v4l stream with
// different encoder types (e.g AV1)

fn fourcc_to_input_type(fourcc: v4l::FourCC) -> Result<InputType> {
    match &fourcc.repr[..] {
        b"BGR4" => Ok(AvPixel::BGR32),
        b"BGR3" => Ok(AvPixel::BGR24),
        b"RGB3" => Ok(AvPixel::RGB24),
        b"YUYV" => Ok(AvPixel::YUYV422),
        b"UYVY" => Ok(AvPixel::UYVY422),
        b"YV12" => Ok(AvPixel::YUV420P),
        b"NV12" => Ok(AvPixel::NV12),
        b"NV21" => Ok(AvPixel::NV21),
        b"MJPG" => Ok(AvPixel::YUVJ420P),
        b"GREY" => Ok(AvPixel::GRAY8),
        _ => bail!("Unsupported v4l FourCC type: {:?}", fourcc),
    }
}

#[derive(Clone)]
pub struct LoadingImage {
    pub data: Vec<u8>,
    pub input_width: u32,
    pub input_height: u32,
    pub input_type: InputType,
}

pub struct V4lH264Config {
    pub output_width: u32,
    pub output_height: u32,
    pub bitrate: usize,
    pub video_dev: String,
    pub v4l_fourcc: v4l::FourCC,
    pub loading_image: Option<LoadingImage>,
}

pub struct V4lH264Stream {}

impl V4lH264Stream {
    pub fn new(
        cfg: V4lH264Config,
        ffmpeg_opts: FfmpegOptions,
    ) -> Result<StreamReader<ReceiverStream<Result<BytesMut, io::Error>>, BytesMut>> {
        let input_type = fourcc_to_input_type(cfg.v4l_fourcc)?;
        // only allow 10 frames to be buffered
        // TODO: maybe make this a configurable option
        let (tx, rx) = mpsc::channel::<Result<BytesMut, io::Error>>(10);

        std::thread::spawn(move || {
            let mut loading_ffmpeg_opts = FfmpegOptions::new();
            loading_ffmpeg_opts.push((String::from("preset"), String::from("medium")));
            loading_ffmpeg_opts.push((String::from("tune"), String::from("stillimage")));
            loading_ffmpeg_opts.push((
                String::from("x264-params"),
                String::from("repeat-headers=1:keyint=1:min-keyint=1:scenecut=0"),
            ));
            // TODO: better error handling, should close the channel correctly instead of exploding
            let cached_loading_frames = cfg.loading_image.as_ref().map(|loading_image| {
                let loading_ec = EncoderConfig {
                    input_width: loading_image.input_width,
                    input_height: loading_image.input_height,
                    output_width: cfg.output_width,
                    output_height: cfg.output_height,
                    framerate: 10,
                    gop: Some(1),
                    bitrate: 1000000,
                    disable_b_frames: true,
                    enc_type: EncoderType::X264,
                    input_type: loading_image.input_type,
                };

                let mut f = File::create("loading-video.h264").unwrap();

                let mut loading_encoder =
                    VideoEncoder::new(loading_ec, &loading_ffmpeg_opts).unwrap();
                let mut nal_frames = Vec::new();
                loop {
                    if let Some(encoded_frame) = loading_encoder
                        .encode_raw(Some(0), &loading_image.data)
                        .unwrap()
                    {
                        nal_frames.push(encoded_frame.data.freeze());
                        break;
                    }
                }

                for encoded_frame in loading_encoder.drain().unwrap() {
                    let frame = encoded_frame.data.clone().freeze();
                    f.write_all(&frame);
                    f.flush();
                    nal_frames.push(frame);
                }

                tracing::error!("Total Loaded Image NAL frames: {}", nal_frames.len());

                nal_frames
            });

            // Open the capture device and wait until the upstream writer has
            // negotiated the requested FourCC. Every failure path retries with
            // the loading image instead of panicking: consumers build with
            // `panic = "abort"`, so a panic here aborts the whole process, and
            // with a systemd `Restart=on-failure` unit that turns a transient
            // condition into an unrecoverable crash loop (CRSW-402).
            let (mut v4l_dev, format) = 'open: loop {
                match Device::with_path(&cfg.video_dev) {
                    Ok(dev) => match dev.format() {
                        Ok(fmt) if fmt.fourcc == cfg.v4l_fourcc => break 'open (dev, fmt),
                        Ok(fmt) => tracing::error!(
                            "{} advertises FourCC {} but {} was requested (writer not ready?)",
                            &cfg.video_dev.as_str(),
                            fmt.fourcc,
                            cfg.v4l_fourcc
                        ),
                        Err(e) => tracing::error!(
                            "Failed to read format on {}: {e}",
                            &cfg.video_dev.as_str()
                        ),
                    },
                    Err(e) => {
                        tracing::error!("Failed to open {}: {e}", &cfg.video_dev.as_str())
                    }
                }
                if let Some(frames) = cached_loading_frames.as_ref() {
                    if !send_loading_frames(&tx, frames) {
                        return;
                    }
                } else {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            };

            // Allocate the mmap capture ring. VIDIOC_REQBUFS returns EBUSY when
            // another consumer still holds the (shared) v4l2loopback buffer
            // pool. Retry with the loading image and a fresh fd rather than
            // panicking, so the reader self-heals once the pool frees up
            // instead of crash-looping the process (CRSW-402).
            let mut stream = 'reqbufs: loop {
                match MmapStream::new(&v4l_dev, Type::VideoCapture) {
                    Ok(s) => break 'reqbufs s,
                    Err(e) => {
                        tracing::error!(
                            "VIDIOC_REQBUFS on {} failed ({e}); another consumer may still \
                             hold the buffer pool. Retrying…",
                            &cfg.video_dev.as_str()
                        );
                        if let Some(frames) = cached_loading_frames.as_ref() {
                            if !send_loading_frames(&tx, frames) {
                                return;
                            }
                        } else {
                            std::thread::sleep(std::time::Duration::from_secs(1));
                        }
                        // Reopen so the kernel-side fd state is fresh before the
                        // next REQBUFS attempt.
                        match Device::with_path(&cfg.video_dev) {
                            Ok(dev) => v4l_dev = dev,
                            Err(e) => tracing::error!(
                                "Failed to reopen {}: {e}",
                                &cfg.video_dev.as_str()
                            ),
                        }
                    }
                }
            };

            debug!("V4L Format: {:?}", format);

            // Track the geometry the mmap ring was allocated for. When the
            // upstream writer renegotiates the V4L2 format mid-stream (e.g.
            // a process swap from torchyd2 to gtd, with different output
            // dimensions), v4l_dev.format() begins reporting the new dims
            // but the existing MmapStream's buffers are still sized for the
            // old ones — frames land partially-filled, rows misaligned, the
            // tail black. Detect that and rebuild the stream below.
            let mut active_width = format.width;
            let mut active_height = format.height;
            // TODO: Make this EncoderConfig settable by the user
            let ec = EncoderConfig {
                input_width: format.width,
                input_height: format.height,
                output_width: cfg.output_width,
                output_height: cfg.output_height,
                framerate: 15,
                gop: None,
                bitrate: cfg.bitrate,
                disable_b_frames: true,
                enc_type: EncoderType::X264,
                input_type,
            };

            // Add repeat-headers=1 to ensure SPS/PPS are emitted regularly.
            // This is critical for seamless transitions from overlay to live feed,
            // as the decoder needs fresh SPS/PPS headers to reinitialize.
            let mut live_ffmpeg_opts = ffmpeg_opts.clone();
            live_ffmpeg_opts.push((
                String::from("x264-params"),
                String::from("repeat-headers=1"),
            ));

            let mut pts: i64 = 0;
            let mut encoder = VideoEncoder::new(ec, &live_ffmpeg_opts).unwrap();

            // Bytes-per-pixel for the configured input. BGR3/RGB3 are 3;
            // single-plane Mono formats would be 1. v4l2loopback frames are
            // always single-plane packed so this is correct as a divisor.
            let bpp: usize = match input_type {
                AvPixel::BGR24 | AvPixel::RGB24 => 3,
                AvPixel::GRAY8 => 1,
                AvPixel::RGBA | AvPixel::BGRA | AvPixel::ARGB | AvPixel::ABGR => 4,
                _ => 3,
            };

            loop {
                // TODO: Better error handling
                match stream.next() {
                    Ok((m_buf, meta)) => {
                        let bytesused = meta.bytesused as usize;
                        // debug!("V4L bytesused: {}", meta.bytesused);

                        // Detect geometry renegotiation by the upstream
                        // writer. v4l_dev.format() is unreliable as a signal
                        // here — v4l2loopback caches the format the *first*
                        // writer set and reports that forever, even after a
                        // later writer renegotiates. The actual frame size
                        // is bytesused; trust that over the format query.
                        //
                        // When the bytes-per-frame doesn't match the dims
                        // we're currently driving the encoder with, the
                        // writer changed. Reopen the device (forces a fresh
                        // kernel-side format read) and rebuild the mmap
                        // ring at the new geometry. Drop this frame — the
                        // next dequeue lands on a correctly-sized buffer.
                        let expected_bytes = active_width as usize * active_height as usize * bpp;
                        if bytesused != expected_bytes && bytesused > 0 {
                            // Derive new dims. Width is the most reliable
                            // single value to read from format() since rows
                            // are stride-packed; total bytes / (width * bpp)
                            // gives height regardless of whether the format
                            // query lied about height.
                            let probe_format = v4l_dev.format().unwrap();
                            let new_width = probe_format.width;
                            let derived_height = if new_width > 0 {
                                (bytesused / (new_width as usize * bpp)) as u32
                            } else {
                                0
                            };

                            debug!(
                                "V4L frame size changed: {}x{} ({} bytes) -> {}x{} ({} bytes); \
                                 reopening device and rebuilding mmap ring",
                                active_width, active_height, expected_bytes,
                                new_width, derived_height, bytesused
                            );

                            drop(stream);
                            // Reopen so the kernel-side state for this fd
                            // is fresh — v4l2loopback ties cached format to
                            // the open fd.
                            v4l_dev = Device::with_path(&cfg.video_dev)
                                .expect("Failed to reopen v4l device");
                            stream = MmapStream::new(&v4l_dev, Type::VideoCapture).unwrap();
                            active_width = new_width;
                            active_height = derived_height;
                            // Skip this stale frame; the next iteration
                            // pulls a fresh one at the new geometry.
                            continue;
                        }

                        // encode_raw_sized rebuilds the scaler when the dims
                        // change and returns a typed error rather than
                        // panicking inside copy_from_slice.
                        if let Some(encoded_frame) = encoder
                            .encode_raw_sized(
                                Some(pts),
                                &m_buf[..bytesused],
                                active_width,
                                active_height,
                            )
                            .unwrap()
                        {
                            tx.blocking_send(Ok(encoded_frame.data)).unwrap();
                        }
                        pts += 1;
                    }
                    Err(e) => {
                        if let Some(error_code) = e.raw_os_error() {
                            if error_code == 5 {
                                error!(
                                    "Got I/O Error: {} for {}. Retrying in 1 second",
                                    error_code, &cfg.video_dev
                                );
                                if let Some(frames) = cached_loading_frames.as_ref() {
                                    if !send_loading_frames(&tx, frames) {
                                        return;
                                    }
                                } else {
                                    std::thread::sleep(std::time::Duration::from_secs(1));
                                }
                            } else {
                                panic!("Unrecoverable OS Error: {}", e);
                            }
                        }
                    }
                }
            }
        });

        Ok(StreamReader::new(ReceiverStream::new(rx)))
    }
}

fn send_loading_frames(tx: &mpsc::Sender<Result<BytesMut, io::Error>>, frames: &[Bytes]) -> bool {
    if frames.is_empty() {
        return true;
    }

    for frame in frames {
        if tx
            .blocking_send(Ok(BytesMut::from(frame.as_ref())))
            .is_err()
        {
            return false;
        }

        std::thread::sleep(std::time::Duration::from_millis(250));
    }

    true
}
