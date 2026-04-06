import os
import subprocess

source_folder = 'videos'
output_folder = 'audios'
ffmpeg_path = r"C:\Users\sarth\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

# Create the output folder if it doesn't exist yet
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of files
all_videos = os.listdir(source_folder)

"""function to convert video to audios using ffmpeg. It takes the path of the video and the desired path for the audios output."""
def convert_video_to_audio(video_path, audio_path):
    # Check if the video path exists
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return
    
    command = [ffmpeg_path, '-y', '-i', video_path, '-vn', audio_path]

    try:
        # capture_output=True hides the massive wall of text FFmpeg prints
        subprocess.run(command, check=True, capture_output=True)
        print(f"Successfully converted: {video_path} -> {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {video_path}: {e}")

# Loop through the videos
for video in all_videos:
    # we separate the filename from the extension.
    filename_no_ext, extension = os.path.splitext(video)
    
    # remove parts after '(', we do it safely:
    if '(' in filename_no_ext:
        name = filename_no_ext.split('(')[0].strip()
    else:
        name = filename_no_ext.strip()

    # 4. Use os.path.join for paths
    # This ensures the slashes (/ or \) are correct for Windows/Mac/Linux
    input_path = os.path.join(source_folder, video)
    output_path = os.path.join(output_folder, f'{name}.mp3')
    
    convert_video_to_audio(input_path, output_path)