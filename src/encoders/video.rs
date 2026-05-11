use ffmpeg::codec::codec::Codec as AvCodec;
use ffmpeg::codec::context::Context as AvContext;
use ffmpeg::codec::encoder::video::Encoder as AvVideoEncoder;
use ffmpeg::codec::packet::Packet as AvPacket;
use ffmpeg::software::scaling::{context::Context as AvScalingContext, flag::Flags};
use ffmpeg::util::format::Pixel as AvPixel;
use ffmpeg::util::frame::Video as AvFrame;
use ffmpeg::util::rational::Rational;
use ffmpeg::Dictionary as AvDictionary;
use ffmpeg::Error as AvError;
use ffmpeg_next as ffmpeg;

#[cfg(feature = "opencv")]
use opencv::{
    core::Mat,
    prelude::{MatTraitConst, MatTraitConstManual},
};

use serde::{Deserialize, Serialize};
use strum::{Display, EnumString, IntoStaticStr};

use thiserror::Error;

use tracing::{debug, error};

// pub type FfmpegOptions<'a> = AvDictionary<'a>;
// NOTE: does this have to be a tuple of Strings? Maybe CoW can help
pub type FfmpegOptions = Vec<(String, String)>;
pub type InputType = AvPixel;

/// The video codec standard, independent of the specific encoder implementation.
#[derive(Serialize, Deserialize, Debug, Display, EnumString, IntoStaticStr, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    #[serde(rename = "h264")]
    #[strum(serialize = "h264")]
    H264,
    #[serde(rename = "av1")]
    #[strum(serialize = "av1")]
    Av1,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EncoderType {
    // H.264
    #[serde(rename = "libx264")]
    X264,
    #[serde(rename = "h264_nvenc")]
    H264Nvenc,
    #[serde(rename = "h264_nvmpi")]
    // NOTE: Only for Jetson NVENC systems. Will not work on any other
    // systems
    H264Nvmpi,
    // AV1
    #[serde(rename = "libsvtav1")]
    SvtAv1,
    #[serde(rename = "av1_nvenc")]
    Av1Nvenc,
}

impl EncoderType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::X264 => "libx264",
            Self::H264Nvenc => "h264_nvenc",
            Self::H264Nvmpi => "h264_nvmpi",
            Self::SvtAv1 => "libsvtav1",
            Self::Av1Nvenc => "av1_nvenc",
        }
    }

    /// Returns the video codec standard for this encoder.
    pub fn codec(&self) -> VideoCodec {
        match self {
            Self::X264 | Self::H264Nvenc | Self::H264Nvmpi => VideoCodec::H264,
            Self::SvtAv1 | Self::Av1Nvenc => VideoCodec::Av1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub framerate: u32,
    pub gop: Option<u32>,
    pub bitrate: usize,
    pub disable_b_frames: bool,
    pub enc_type: EncoderType,
    pub input_type: InputType,
}

#[derive(Error, Debug)]
pub enum VideoEncoderError {
    #[error("PTS was not monotonically increasing: previous PTS: {prev_pts} > PTS: {curr_pts}")]
    PTSNotMonotonic {
        prev_pts: i64,
        curr_pts: i64
    },
    #[error("input slice length {actual} does not match expected {expected} bytes for {width}x{height} {input_type:?}")]
    InputSizeMismatch {
        actual: usize,
        expected: usize,
        width: u32,
        height: u32,
        input_type: AvPixel,
    },
    #[error("ffmpeg error: ")]
    FfmpegError(#[from] AvError),
    #[error("failed to alloc avcodec context")]
    AvCodecAllocContextError,
}

pub struct VideoEncoder {
    encoder: AvVideoEncoder,
    scaler: AvScalingContext,
    prev_pts: Option<i64>,
    input_type: AvPixel,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
}

fn codec_context_as(codec: &AvCodec) -> Result<AvContext, VideoEncoderError> {
    unsafe {
        let context_ptr = ffmpeg::ffi::avcodec_alloc_context3(codec.as_ptr());
        if !context_ptr.is_null() {
            Ok(AvContext::wrap(context_ptr, None))
        } else {
            Err(VideoEncoderError::AvCodecAllocContextError)
        }
    }
}

pub struct EncodedFrame {
    pub data: bytes::BytesMut,
    pub is_keyframe: bool,
    pub duration: i64,
    pub pts: Option<i64>,
}

unsafe impl Send for VideoEncoder {}
unsafe impl Sync for VideoEncoder {}

impl VideoEncoder {
    pub fn new(
        ec: EncoderConfig,
        extra_ffmpeg_opts: &FfmpegOptions,
    ) -> Result<Self, VideoEncoderError> {
        let mut extra_opts = AvDictionary::new();
        for opt in extra_ffmpeg_opts {
            extra_opts.set(opt.0.as_str(), opt.1.as_str());
        }

        let ffmpeg_codec = ffmpeg::encoder::find_by_name(ec.enc_type.as_str()).unwrap();
        let ffmpeg_context = codec_context_as(&ffmpeg_codec)?;
        let mut ffmpeg_vid_encoder = ffmpeg_context.encoder().video().unwrap();
        ffmpeg_vid_encoder.set_width(ec.output_width);
        ffmpeg_vid_encoder.set_height(ec.output_height);
        ffmpeg_vid_encoder.set_format(AvPixel::YUV420P);
        ffmpeg_vid_encoder.set_frame_rate(Some((ec.framerate as i32, 1)));
        ffmpeg_vid_encoder.set_time_base(Rational(1, ec.framerate as i32));
        ffmpeg_vid_encoder.set_bit_rate(ec.bitrate as usize);
        if ec.disable_b_frames {
            ffmpeg_vid_encoder.set_max_b_frames(0_usize);
        }
        if let Some(gop) = ec.gop {
            ffmpeg_vid_encoder.set_gop(gop);
        }
        let encoder = ffmpeg_vid_encoder.open_with(extra_opts)?;

        let scaler = AvScalingContext::get(
            ec.input_type,
            ec.input_width,
            ec.input_height,
            AvPixel::YUV420P,
            ec.output_width,
            ec.output_height,
            Flags::BILINEAR,
        )?;

        Ok(Self {
            encoder,
            scaler,
            prev_pts: None,
            input_type: ec.input_type,
            input_width: ec.input_width,
            input_height: ec.input_height,
            output_width: ec.output_width,
            output_height: ec.output_height,
        })
    }

    #[cfg(feature = "opencv")]
    pub fn encode_mat(&mut self, pts: Option<i64>, input: &Mat) -> Result<Option<EncodedFrame>, VideoEncoderError> {
        let width = input.cols();
        let height = input.rows();
        let mut out_frame = AvFrame::new(
            AvPixel::YUV420P,
            width.try_into().unwrap(),
            height.try_into().unwrap(),
        );

        let mut in_frame = AvFrame::new(
            self.input_type,
            width.try_into().unwrap(),
            height.try_into().unwrap(),
        );

        in_frame
            .data_mut(0)
            .copy_from_slice(input.data_bytes().unwrap());

        self.scaler.run(&in_frame, &mut out_frame).unwrap();
        self.encode(pts, out_frame)
    }

    /// Encode a raw byte slice at the dimensions configured when the
    /// [`VideoEncoder`] was constructed. Convenience wrapper around
    /// [`Self::encode_raw_sized`]; prefer the sized variant when the source
    /// geometry can vary at runtime (e.g. a V4L2 loopback whose writer was
    /// swapped without re-initializing the reader).
    pub fn encode_raw(&mut self, pts: Option<i64>, input: &[u8]) -> Result<Option<EncodedFrame>, VideoEncoderError> {
        let w = self.input_width;
        let h = self.input_height;
        self.encode_raw_sized(pts, input, w, h)
    }

    /// Encode a raw byte slice whose dimensions may differ from the
    /// `input_width`/`input_height` the encoder was constructed with.
    /// Rebuilds the input frame and the scaler whenever `(width, height)`
    /// changes, so it's safe to drive from a source that renegotiates
    /// geometry on the fly.
    ///
    /// Output dimensions stay locked to whatever was configured in
    /// [`EncoderConfig`], so the encoded stream remains consistent across
    /// input-size flips.
    ///
    /// Returns [`VideoEncoderError::InputSizeMismatch`] when `input.len()`
    /// does not match the bytes implied by `width`, `height`, and the
    /// configured `input_type`.
    pub fn encode_raw_sized(
        &mut self,
        pts: Option<i64>,
        input: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Option<EncodedFrame>, VideoEncoderError> {
        debug!("input len: {} ({}x{})", input.len(), width, height);

        if input.is_empty() {
            debug!("Received empty frame, skipping encoding");
            return Ok(None);
        }

        // Rebuild scaler when input geometry changes from the previous call.
        // Allocation is light; this only fires on writer swaps in practice.
        if width != self.input_width || height != self.input_height {
            debug!(
                "input geometry changed: {}x{} -> {}x{}; rebuilding scaler",
                self.input_width, self.input_height, width, height
            );
            self.scaler = AvScalingContext::get(
                self.input_type,
                width,
                height,
                AvPixel::YUV420P,
                self.output_width,
                self.output_height,
                Flags::BILINEAR,
            )?;
            self.input_width = width;
            self.input_height = height;
        }

        let mut out_frame = AvFrame::new(
            AvPixel::YUV420P,
            self.output_width.try_into().unwrap(),
            self.output_height.try_into().unwrap(),
        );

        let mut in_frame = AvFrame::new(
            self.input_type,
            width.try_into().unwrap(),
            height.try_into().unwrap(),
        );

        // Validate the slice length matches the frame buffer before copying.
        // Raises a typed error instead of panicking inside copy_from_slice.
        let expected: usize = (0..in_frame.planes()).map(|i| in_frame.data(i).len()).sum();
        if input.len() != expected {
            return Err(VideoEncoderError::InputSizeMismatch {
                actual: input.len(),
                expected,
                width,
                height,
                input_type: self.input_type,
            });
        }

        if in_frame.planes() > 1 {
            let mut start_id = 0;
            for i in 0..in_frame.planes() {
                let buf_size = in_frame.data(i).len();
                let end_id = start_id + buf_size;
                in_frame.data_mut(i).copy_from_slice(&input[start_id..end_id]);
                start_id = end_id;
            }
        } else {
            in_frame.data_mut(0).copy_from_slice(input);
        }
        self.scaler.run(&in_frame, &mut out_frame).unwrap();
        self.encode(pts, out_frame)
    }

    pub fn encode(&mut self, pts: Option<i64>, mut frame: AvFrame) -> Result<Option<EncodedFrame>, VideoEncoderError> {
        if let Some(prev_pts) = self.prev_pts {
            if let Some(curr_pts) = pts {
                if prev_pts > curr_pts {
                    return Err(VideoEncoderError::PTSNotMonotonic {
                        prev_pts,
                        curr_pts
                    })
                }
            }
        }

        frame.set_pts(pts);
        self.prev_pts = pts;
        self.encoder.send_frame(&frame).unwrap();
        self.retrieve_nal()
    }

    /// Drains and consumes this encoder
    pub fn drain(mut self) -> Result<Vec<EncodedFrame>, VideoEncoderError> {
        self.encoder.send_eof()?;
        let mut result = vec![];
        while let Some(nal) = self.retrieve_nal()? {
            result.push(nal);
        }
        Ok(result)
    }

    /// Drain a NAL out of the encoder. Typically not used directly,
    /// except after `begin_drain`.
    fn retrieve_nal(&mut self) -> Result<Option<EncodedFrame>, VideoEncoderError> {
        let mut encoded_packet = AvPacket::empty();
        let encoder_res = self.encoder.receive_packet(&mut encoded_packet);

        match encoder_res {
            Ok(_) => {
                let encoded_data = encoded_packet.data();
                if encoded_data.is_none() {
                    error!("encoder likely dropping frames!! data packet is empty");
                    Ok(None)
                } else {
                    if let Some(nal) = encoded_data {
                        Ok(Some(EncodedFrame {
                            data: bytes::BytesMut::from(nal),
                            is_keyframe: encoded_packet.is_key(),
                            duration: encoded_packet.duration(),
                            pts: encoded_packet.pts(),
                        }))
                    } else {
                        Ok(None)
                    }
                }
            }
            Err(e) => {
                debug!("got ffmpeg encoder error: {e}");
                // TODO: better error handling here
                Ok(None)
            }
        }
    }
}
