import language_tool_python
import requests
from tqdm import tqdm
from vosk import Model, KaldiRecognizer
from yt_dlp import YoutubeDL
from fpdf import FPDF
import wave
import sys
import os
import json
import subprocess
import shutil

ARCHITECTURE = "linux"  # windows or linux or mac

LANGUAGE = 'en-US'  # fr OR en-US
VOSK_MODEL_VERSION = "vosk-model-en-us-0.22-lgraph"  # vosk-model-fr-0.22 OR vosk-model-en-us-0.22-lgraph
VOSK_REPO_URL = "https://alphacephei.com/vosk/models/"

# FFMPEG Windows sources
FFMPEG_WIN_VERSION = "ffmpeg-5.0-essentials_build"
FFMPEG_WIN_EXTENSION = "zip"
FFMPEG_WIN_REPO_URL = "https://www.gyan.dev/ffmpeg/builds/packages/"

# FFMPEG Linux sources
FFMPEG_LINUX_VERSION = "ffmpeg-5.0-amd64-static"
FFMPEG_LINUX_EXTENSION = "tar.xz"
FFMPEG_LINUX_REPO_URL = "https://johnvansickle.com/ffmpeg/releases/"

# FFMPEG MacOS sources
FFMPEG_MAC_VERSION = "ffmpeg-5.0"
FFMPEG_MAC_EXTENSION = "zip"
FFMPEG_MAC_REPO_URL = "https://evermeet.cx/ffmpeg/"

FFMPEG_EXE_PATH = os.path.join(os.path.dirname(__file__), 'ffmpeg', 'bin', 'ffmpeg') if ARCHITECTURE == 'windows' \
    else os.path.join(os.path.dirname(__file__), 'ffmpeg', 'ffmpeg')


def download_and_unpack_sources(repo_url, file_name, target_dir_name, extension="zip"):

    url = f'{repo_url}{file_name}.{extension}'

    r = requests.get(url, allow_redirects=True)

    total_size_in_bytes = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(f'{file_name}.{extension}', 'wb') as file:
        for data in r.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    shutil.unpack_archive(f'{file_name}.{extension}')

    if target_dir_name:
        os.rename(file_name, target_dir_name)

    os.remove(f'{file_name}.{extension}')


def download_youtube_audio(youtube_url):
    ydl_opts = {
        'ffmpeg_location': FFMPEG_EXE_PATH,
        'outtmpl': 'youtube_dl_input' + '.%(ext)s',
        'format': 'bestaudio/best',
        'restrictfilenames': True,
        'noplaylist': True,
        'postprocessors': [
            {'key': 'FFmpegExtractAudio',
             'preferredcodec': 'wav',
             'preferredquality': '128',
             },
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    subprocess.call([FFMPEG_EXE_PATH, '-loglevel', 'quiet', '-i',
                     'youtube_dl_input.wav',
                     '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                     'youtube_dl_input_mono.wav'])

    os.remove('youtube_dl_input.wav')


def sample_video_as_wav(input_file, sample_rate):
    return subprocess.Popen([FFMPEG_EXE_PATH, '-loglevel', 'quiet', '-i',
                             input_file,
                             '-ar', str(sample_rate), '-ac', '1', '-f', 's16le', '-'],
                            stdout=subprocess.PIPE)


def transcript_file(input_file, model_path, is_audio=True):
    # Check if file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(os.path.basename(input_file) + " not found")

        # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(os.path.basename(model_path) + " not found")

    if is_audio:
        # open audio file
        wf = wave.open(input_file, "rb")

        # check if wave file has the right properties
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise TypeError("Audio file must be WAV format mono PCM.")

        sample_rate = wf.getframerate()

    else:
        sample_rate = 16000

    # Initialize model
    model = Model(model_path)
    rec = KaldiRecognizer(model, sample_rate)

    # Get file size (to calculate progress bar)
    if is_audio:
        file_size = os.path.getsize(input_file)
    else:
        file_size = len(sample_video_as_wav(input_file, sample_rate).stdout.read())
        # Reinit process stdout to the beginning because seek is not possible with stdio
        process = sample_video_as_wav(input_file, sample_rate)

    # Run transcription
    pbar = tqdm(total=file_size)

    # To store our results
    transcription = []

    while True:
        if is_audio:
            data = wf.readframes(4000)  # use buffer of 4000
        else:
            data = process.stdout.read(4000)
        pbar.update(len(data))
        if len(data) == 0:
            pbar.set_description("Transcription finished")
            break
        if rec.AcceptWaveform(data):
            # Convert json output to dict
            result_dict = json.loads(rec.Result())
            # Extract text values and append them to transcription list
            transcription.append(result_dict.get("text", ""))

    # Get final bits of audio and flush the pipeline
    final_result = json.loads(rec.FinalResult())
    transcription.append(final_result.get("text", ""))

    transcription_text = '. '.join(transcription)

    return transcription_text


print(f'Downloading sources for {ARCHITECTURE.upper()}, language {LANGUAGE.upper()}...')

if not os.path.exists("ffmpeg"):
    if ARCHITECTURE == "windows":
        download_and_unpack_sources(FFMPEG_WIN_REPO_URL, FFMPEG_WIN_VERSION, "ffmpeg", FFMPEG_WIN_EXTENSION)
    elif ARCHITECTURE == "linux":
        download_and_unpack_sources(FFMPEG_LINUX_REPO_URL, FFMPEG_LINUX_VERSION, "ffmpeg", FFMPEG_LINUX_EXTENSION)
    else:
        download_and_unpack_sources(FFMPEG_MAC_REPO_URL, FFMPEG_MAC_VERSION, False, FFMPEG_MAC_EXTENSION)
        os.rename("ffmpeg", "ffmpeg_tmp")
        os.mkdir("ffmpeg")
        shutil.move("ffmpeg_tmp", "ffmpeg" + os.sep + "ffmpeg")


if not os.path.exists(f'model-{LANGUAGE}'):
    download_and_unpack_sources(VOSK_REPO_URL, VOSK_MODEL_VERSION, f'model-{LANGUAGE}', extension="zip")

if len(sys.argv) > 1 and sys.argv[1][:4] == "http":
    download_youtube_audio(sys.argv[1])
    transcription = transcript_file("youtube_dl_input_mono.wav", f'model-{LANGUAGE}', is_audio=True)
    os.remove("youtube_dl_input_mono.wav")

elif len(sys.argv) > 1 and sys.argv[1][-3:] == "wav":
    transcription = transcript_file(sys.argv[1], f'model-{LANGUAGE}', is_audio=True)
elif len(sys.argv) > 1:
    transcription = transcript_file(sys.argv[1], f'model-{LANGUAGE}', is_audio=False)
else:
    # Test purpose content
    download_youtube_audio('https://youtu.be/hORwj7g_GFE')
    transcription = transcript_file("youtube_dl_input_mono.wav", f'model-{LANGUAGE}', is_audio=True)
    os.remove("youtube_dl_input_mono.wav")

print("Proceeding basic grammar check...")

tool = language_tool_python.LanguageTool(LANGUAGE)
matches = tool.check(transcription)
transcription_corrected = tool.correct(transcription)

tool.close()

pdf = FPDF()

pdf.add_page()

margin_bottom_mm = 10
pdf.set_auto_page_break(True, margin=margin_bottom_mm)
pdf.set_font("Times", size=12)

words = transcription_corrected.split()
grouped_words = [' '.join(words[i: i + 13]) for i in range(0, len(words), 13)]
for x in grouped_words:
    pdf.cell(50, 10, txt=x, ln=1, align='L')

pdf.output("transcript.pdf")
