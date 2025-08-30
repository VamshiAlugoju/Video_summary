import asyncio
import errno
import fractions
import logging
import threading
import time
from typing import Any, Optional, Union, cast

import av
import av.container
from av import AudioFrame, VideoFrame
from av.audio import AudioStream
from av.frame import Frame
from av.packet import Packet
from av.video.stream import VideoStream

from aiortc.mediastreams import AUDIO_PTIME, MediaStreamError, MediaStreamTrack

logger = logging.getLogger(__name__)

import os
ROOT = os.path.dirname(__file__)


REAL_TIME_FORMATS = [
    "alsa",
    "android_camera",
    "avfoundation",
    "bktr",
    "decklink",
    "dshow",
    "fbdev",
    "gdigrab",
    "iec61883",
    "jack",
    "kmsgrab",
    "openal",
    "oss",
    "pulse",
    "sndio",
    "rtsp",
    "v4l2",
    "vfwcap",
    "x11grab",
]

_AudioOrVideoStream = Union[AudioStream, VideoStream]

 

def player_worker_decode(
    loop: asyncio.AbstractEventLoop,
    container: av.container.InputContainer,
    streams: list[_AudioOrVideoStream],
    audio_track: "PlayerStreamTrack",
    video_track: "PlayerStreamTrack",
    quit_event: threading.Event,
    throttle_playback: bool,
    loop_playback: bool,
) -> None:
    audio_sample_rate = 48000
    audio_samples = 0
    audio_time_base = fractions.Fraction(1, audio_sample_rate)
    audio_resampler = av.AudioResampler(
        format="s16",
        layout="stereo",
        rate=audio_sample_rate,
        frame_size=int(audio_sample_rate * AUDIO_PTIME),
    )

    video_first_pts = None

    frame_time = None
    start_time = time.time()

    while not quit_event.is_set():
        try:
            frame = next(container.decode(*streams))
        except Exception as exc:
            if isinstance(exc, av.FFmpegError) and exc.errno == errno.EAGAIN:
                time.sleep(0.01)
                continue
            if isinstance(exc, StopIteration) and loop_playback:
                container.seek(0)
                continue
            if audio_track:
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(None), loop)
            if video_track:
                asyncio.run_coroutine_threadsafe(video_track._queue.put(None), loop)
            break

        # read up to 1 second ahead
        if throttle_playback:
            elapsed_time = time.time() - start_time
            if frame_time and frame_time > elapsed_time + 1:
                time.sleep(0.1)

        if isinstance(frame, AudioFrame) and audio_track:
            for frame in audio_resampler.resample(frame):
                # fix timestamps
                frame.pts = audio_samples
                frame.time_base = audio_time_base
                audio_samples += frame.samples

                frame_time = frame.time
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(frame), loop)
        elif isinstance(frame, VideoFrame) and video_track:
            if frame.pts is None:  # pragma: no cover
                logger.warning(
                    "AudioPlayer(%s) Skipping video frame with no pts", container.name
                )
                continue

            # video from a webcam doesn't start at pts 0, cancel out offset
            if video_first_pts is None:
                video_first_pts = frame.pts
            frame.pts -= video_first_pts

            frame_time = frame.time
            asyncio.run_coroutine_threadsafe(video_track._queue.put(frame), loop)


def player_worker_demux(
    loop: asyncio.AbstractEventLoop,
    container: av.container.InputContainer,
    streams: list[_AudioOrVideoStream],
    audio_track: "PlayerStreamTrack",
    video_track: "PlayerStreamTrack",
    quit_event: threading.Event,
    throttle_playback: bool,
    loop_playback: bool,
) -> None:
    video_first_pts = None
    frame_time = None
    start_time = time.time()

    while not quit_event.is_set():
        try:
            packet = next(container.demux(*streams))
            if not packet.size:
                raise StopIteration
        except Exception as exc:
            if isinstance(exc, av.FFmpegError) and exc.errno == errno.EAGAIN:
                time.sleep(0.01)
                continue
            if isinstance(exc, StopIteration) and loop_playback:
                container.seek(0)
                continue
            if audio_track:
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(None), loop)
            if video_track:
                asyncio.run_coroutine_threadsafe(video_track._queue.put(None), loop)
            break

        # read up to 1 second ahead
        if throttle_playback:
            elapsed_time = time.time() - start_time
            if frame_time and frame_time > elapsed_time + 1:
                time.sleep(0.1)

        track = None
        if isinstance(packet.stream, AudioStream) and audio_track:
            track = audio_track
        elif isinstance(packet.stream, VideoStream) and video_track:
            if packet.pts is None:  # pragma: no cover
                logger.warning(
                    "AudioPlayer(%s) Skipping video packet with no pts", container.name
                )
                continue
            track = video_track

            # video from a webcam doesn't start at pts 0, cancel out offset
            if video_first_pts is None:
                video_first_pts = packet.pts
            packet.pts -= video_first_pts

        if (
            track is not None
            and packet.pts is not None
            and packet.time_base is not None
        ):
            frame_time = int(packet.pts * packet.time_base)
            asyncio.run_coroutine_threadsafe(track._queue.put(packet), loop)

 
class PlayerStreamTrack(MediaStreamTrack):
    def __init__(self, player: "AudioPlayer", kind: str) -> None:
        super().__init__()
        self.kind = kind
        self._player: Optional[AudioPlayer] = player
        self._queue: asyncio.Queue[Union[Frame, Packet]] = asyncio.Queue()
        self._start: Optional[float] = None

 
    
    async def recv(self) -> Union[Frame, Packet]:
      
        if self.readyState != "live":
            raise MediaStreamError

        self._player._start(self)
        data = await self._queue.get()
        # print("size of queue :" , self._queue.qsize())
        if data is None :
            self.stop()
            raise MediaStreamError
        if isinstance(data, Frame): 
            data_time = data.time
        elif isinstance(data, Packet):
            data_time = float(data.pts * data.time_base)

        # control playback rate
        if (
            self._player is not None
            and self._player._throttle_playback
            and data_time is not None
        ):
            if self._start is None:
                self._start = time.time() - data_time
            else:
                wait = self._start + data_time - time.time()
                await asyncio.sleep(wait)

        return data

    def stop(self) -> None:
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None


class AudioPlayer:
    """
    A media source that reads audio from a file.

    Examples:

    .. code-block:: python

        # Open a video file.
        player = AudioPlayer('/path/to/some.mp4')

        # Open an HTTP stream.
        player = AudioPlayer(
            'http://download.tsi.telecom-paristech.fr/'
            'gpac/dataset/dash/uhd/mux_sources/hevcds_720p30_2M.mp4')

    :param file: The path to a file, or a file-like object.
    :param format: The format to use, defaults to autodect.
    :param options: Additional options to pass to FFmpeg.
    :param timeout: Open/read timeout to pass to FFmpeg.
    :param loop: Whether to repeat playback indefinitely (requires a seekable file).
    """

    def __init__(
        self,
        file: Any,
        format: Optional[str] = None,
        options: Optional[dict[str, str]] = None,
        timeout: Optional[int] = None,
        loop: bool = False,
        decode: bool = True,
        on_ended :Any = None,
        sid : Any = None,
    ) -> None:
        self.__container = av.open(
            file=file, format=format, mode="r", options=options, timeout=timeout
        )
        self.__file_name = file
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None
        self.__on_ended = on_ended
        # examine streams
        self.__started: set[PlayerStreamTrack] = set()
        self.__streams: list[_AudioOrVideoStream] = []
        self.__decode = decode
        self.__audio: Optional[PlayerStreamTrack] = None
        self._sid = sid
        for stream in self.__container.streams:
            if stream.type == "audio" and not self.__audio:
                if self.__decode:
                    self.__audio = PlayerStreamTrack(self, kind="audio")
                    self.__streams.append(stream)
                elif stream.codec_context.name in ["opus", "pcm_alaw", "pcm_mulaw"]:
                    self.__audio = PlayerStreamTrack(self, kind="audio")
                    self.__streams.append(stream)

        # check whether we need to throttle playback
        container_format = set(self.__container.format.name.split(","))
        self._throttle_playback = not container_format.intersection(REAL_TIME_FORMATS)

        # check whether the looping is supported
        assert not loop or self.__container.duration is not None, (
            "The `loop` argument requires a seekable file"
        )
        self._loop_playback = loop
        asyncio.create_task(self.stop_with_deley())
      

    
    async def stop_with_deley(self):
        print(f"playing {self.__file_name}")
        # Get the first audio stream
        audio_stream = next(s for s in self.__container.streams if s.type == 'audio')

        # Duration in seconds
        duration_seconds = float(audio_stream.duration * audio_stream.time_base)
        trimmed_duration = max(0, duration_seconds-1 )
        print(f"Audio duration: {trimmed_duration:.2f} seconds")
        print(f"stopping audio player with deley {trimmed_duration} sec")
        await asyncio.sleep(trimmed_duration)
        print("stopping pc conn")
        await self.__on_ended()        

    @property
    def audio(self) -> Optional[MediaStreamTrack]:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_decode if self.__decode else player_worker_demux,
                args=(
                    asyncio.get_event_loop(),
                    self.__container,
                    self.__streams,
                    self.__audio,
                    None,
                    self.__thread_quit,
                    self._throttle_playback,
                    self._loop_playback,
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)
       
         
        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            self.__container.close()
            self.__container = None

    def __log_debug(self, msg: str, *args: object) -> None:
        logger.debug(f"AudioPlayer(%s) {msg}", self.__container.name, *args)





class PlayerStreamTrackPlayer(MediaStreamTrack):
    kind = "audio"

    def __init__(self , on_ended : Any ,sid : Any ) :
        super().__init__()
        self._queue: asyncio.Queue[Union[Frame, Packet]] = asyncio.Queue()
        self._start: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_quit: Optional[threading.Event] = None
        self._container: Optional[av.container.InputContainer] = None
        self._streams: list[_AudioOrVideoStream] = []
        self._decode: bool = True
        self._throttle_playback: bool = True
        self._loop_playback: bool = False
        self._on_ended = on_ended
        self._sid = sid
        self._loop =   asyncio.get_event_loop()

    def set_source(
        self,
        file: Any,
        format: Optional[str] = None,
        options: Optional[dict[str, str]] = None,
        timeout: Optional[int] = None,
        loop: bool = False,
        decode: bool = True,
    ) -> None:
        """
        Swap in a new audio file as the source for this track.
        """
        # Stop existing thread if running
        if self._thread:
            self._thread_quit.set()
            self._thread.join()
            self._thread = None

        # Open new file
        self._container = av.open(
            file=file, format=format, mode="r", options=options, timeout=timeout
        )
        self._streams = []
        self._decode = decode
        self._loop_playback = loop

        for stream in self._container.streams:
            if stream.type == "audio":
                self._streams.append(stream)
                break

        # Real-time or not?
        container_format = set(self._container.format.name.split(","))
        self._throttle_playback = not container_format.intersection(REAL_TIME_FORMATS)

        # Start new thread
        self._thread_quit = threading.Event()
        self._thread = threading.Thread(
            name="media-player",
            target=player_worker_decode if self._decode else player_worker_demux,
            args=(
                asyncio.get_event_loop(),
                self._container,
                self._streams,
                self,
                None,  # no video
                self._thread_quit,
                self._throttle_playback,
                self._loop_playback,
            ),
        )
        self._thread.start()

    async def recv(self) -> Union[Frame, Packet]:
        if self.readyState != "live":
            raise MediaStreamError

        data = await self._queue.get()
        if data is None:
            print("data is none")
            # if self._on_ended:
            #     self._on_ended(self._sid)
            # instead of ending, just return silence frame
            self._load_next_file()
            return self._silence_frame()

        if isinstance(data, Frame):
            data_time = data.time
        elif isinstance(data, Packet):
            data_time = float(data.pts * data.time_base)

        # control playback rate
        if self._throttle_playback and data_time is not None:
            if self._start is None:
                self._start = time.time() - data_time
            else:
                wait = self._start + data_time - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)

        return data

    async def stop(self) -> None:
        print("stopping track ---------- called ")
        super().stop()
        if self._thread:
            self._thread_quit.set()
            self._thread.join()
            self._thread = None
        if self._container:
            self._container.close()
            self._container = None

    def _silence_frame(self) -> AudioFrame:
        """
        Generate a short silent frame so the track never dies.
        """
        print("silence frame getting ------------ ")
        frame = av.AudioFrame(format="s16", layout="stereo", samples=960)
        for plane in frame.planes:
            plane.update(b"\x00" * plane.buffer_size)
        frame.sample_rate = 48000
        frame.time_base = fractions.Fraction(1, 48000)
        print("silence frame generated")
        return frame
    

    
    def _load_next_file(self):
        try:
            if self._container:
                self._container.close()

            file_path =  os.path.join(ROOT, "sample_song.wav")
            print(f"Reloading {file_path}")

            self._container = av.open(file_path, mode="r")
            self._streams = [s for s in self._container.streams if s.type == "audio"]

            # reset container position
            try:
                self._container.seek(0)
            except Exception as e:
                print("Seek failed, will read from start anyway:", e)

            self._queue = asyncio.Queue()
            self._started = False
            self._start_reader_thread() 
        except Exception as e:
            print("Error loading next file:", e)
    
    def _start_reader_thread(self):
        if self._started:
            return
        self._started = True

        def _reader():
            for packet in self._container.demux(self._streams):
                for frame in packet.decode():
                    asyncio.run_coroutine_threadsafe(
                        self._queue.put(frame), self._loop
                    )
            # push None to signal EOF
            asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)

        import threading
        threading.Thread(target=_reader, daemon=True).start()