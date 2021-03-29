from gym.wrappers import Monitor
import tempfile
import gym
from gym import Wrapper
from gym import error, version, logger
import os, json, numpy as np
from gym.wrappers.monitoring import stats_recorder, video_recorder
from gym.utils import atomic_write, closer
from gym.utils.json_utils import json_encode_np


def every_episode(episode_id):
    return True

class Monitor_Multi(Monitor):
    def __init__(self, env, directory, video_callable=every_episode, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None, camera_names=None):
        super(Monitor, self).__init__(env)

        self.videos = []

        self.stats_recorder = None
        self.video_recorder = None
        self.enabled = False
        self.episode_id = 0
        self._monitor_id = None
        self.env_semantics_autoreset = env.metadata.get('semantics.autoreset')
        self.camera_names = camera_names

        self._start(directory, video_callable, force, resume,
                    write_upon_reset, uid, mode)

    def reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        #
        # TODO: calculate a more correct 'episode_id' upon merge
        self.video_recorder = VideoRecorder_Multi(
            env=self.env,
            base_path=os.path.join(self.directory,
                                   '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled = self._video_enabled(),
            camera_names = self.camera_names
        )
        self.video_recorder.capture_frame()


class VideoRecorder_Multi(video_recorder.VideoRecorder):
    """VideoRecorder renders a nice movie of a rollout, frame by frame. It
        comes with an `enabled` option so you can still use the same code
        on episodes where you don't want to record video.

        Note:
            You are responsible for calling `close` on a created
            VideoRecorder, or else you may leak an encoder process.

        Args:
            env (Env): Environment to take video of.
            path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
            metadata (Optional[dict]): Contents to save to the metadata file.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
        """

    def __init__(self, env, path=None, metadata=None, enabled=True, base_path=None, camera_names=None):
        # super().__init__(env, path, metadata, enabled, base_path)
        modes = env.metadata.get('render.modes', [])
        self._async = env.metadata.get('semantics.async')
        self.enabled = enabled
        if isinstance(camera_names, list):
            self.camera_names = camera_names
        else:
            self.camera_names = []
        self.encoders=[]

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        self.ansi_mode = False
        if 'rgb_array' not in modes:
            if 'ansi' in modes:
                self.ansi_mode = True
            else:
                logger.info(
                    'Disabling video recorder because {} neither supports video mode "rgb_array" nor "ansi".'.format(
                        env))
                # Whoops, turns out we shouldn't be enabled after all
                self.enabled = False
                return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        self.last_frame = None
        self.env = env

        # Path generation
        self.paths = []
        required_ext = '.json' if self.ansi_mode else '.mp4'
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + '_view0' + required_ext
                self.paths.append(path)
                if self.camera_names is not None:
                    for cam in self.camera_names:
                        self.paths.append(base_path + '_' + cam + required_ext)
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext, delete=False) as f:
                    path = f.name
        else:
            self.paths.append(self.path)

        if self.paths:
            for path in self.paths:
                self.path = path

                path_base, actual_ext = os.path.splitext(self.path)

                if actual_ext != required_ext:
                    hint = " HINT: The environment is text-only, therefore we're recording its text output in a structured JSON format." if self.ansi_mode else ''
                    raise error.Error(
                        "Invalid path given: {} -- must have file extension {}.{}".format(self.path, required_ext, hint))
                # Touch the file in any case, so we know it's present. (This
                # corrects for platform platform differences. Using ffmpeg on
                # OS X, the file is precreated, but not on Linux.
                video_recorder.touch(path)

                self.frames_per_sec = env.metadata.get('video.frames_per_second', 30)
                self.output_frames_per_sec = env.metadata.get('video.output_frames_per_second', self.frames_per_sec)
                self.encoder = None  # lazily start the process
                self.broken = False

                # Dump metadata
                self.metadata = metadata or {}
                self.metadata['content_type'] = 'video/vnd.openai.ansivid' if self.ansi_mode else 'video/mp4'
                self.metadata_path = '{}.meta.json'.format(path_base)
                self.write_metadata()

                logger.info('Starting new video recorder writing to %s', self.path)
                self.empty = True

    def close(self):
        """Make sure to manually close, or else you'll leak the encoder process"""
        if not self.enabled:
            return

        if self.encoders:
            logger.debug('Closing video encoder: path=%s', self.paths)
            for encoder in self.encoders:
                encoder.close()
                encoder = None
            self.encoders = []
        else:
            # No frames captured. Set metadata, and remove the empty output file.
            os.remove(self.path)

            if self.metadata is None:
                self.metadata = {}
            self.metadata['empty'] = True

        # If broken, get rid of the output file, otherwise we'd leak it.
        if self.broken:
            logger.info('Cleaning up paths for broken video recorder: path=%s metadata_path=%s', self.path,
                        self.metadata_path)

            # Might have crashed before even starting the output file, don't try to remove in that case.
            if os.path.exists(self.path):
                os.remove(self.path)

            if self.metadata is None:
                self.metadata = {}
            self.metadata['broken'] = True

        self.write_metadata()

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        if self.paths:
            for i, path in enumerate(self.paths):
                self.path = path
                if not self.functional: return
                logger.debug('Capturing video frame: path=%s', self.path)

                if i == 0:
                    render_mode = 'ansi' if self.ansi_mode else 'rgb_array'
                    frame = self.env.render(mode=render_mode, camera_id=-1)
                else:
                    render_mode = 'ansi' if self.ansi_mode else 'rgb_array'
                    frame = self.env.render(mode=render_mode, camera_name=self.camera_names[i-1])

                if frame is None:
                    if self._async:
                        return
                    else:
                        # Indicates a bug in the environment: don't want to raise
                        # an error here.
                        logger.warn('Env returned None on render(). Disabling further rendering for video recorder by marking as disabled: path=%s metadata_path=%s', self.path, self.metadata_path)
                        self.broken = True
                else:
                    self.last_frame = frame
                    if self.ansi_mode:
                        self._encode_ansi_frame(frame)
                    else:
                        self._encode_image_frame(frame, i)

    def _encode_image_frame(self, frame, i=0):
        if self.paths and not self.encoders:
            for path in self.paths:
                self.encoder = video_recorder.ImageEncoder(path, frame.shape, self.frames_per_sec, self.output_frames_per_sec)
                self.metadata['encoder_version'] = self.encoder.version_info
                self.encoders.append(self.encoder)

        try:
            self.encoders[i].capture_frame(frame)
        except error.InvalidFrame as e:
            logger.warn('Tried to pass invalid video frame, marking as broken: %s', e)
            self.broken = True
        else:
            self.empty = False