import subprocess
import imageio_ffmpeg as ffmpeg

# convert mp4 to 20fps 640x360 for faster processing
def convert_video(input_path, output_path):
    command = [
        ffmpeg.get_ffmpeg_exe(),
        '-i', input_path,
        '-vf', 'fps=20,scale=640:360',
        '-c:v', 'libx264',
        '-crf', '23',
        output_path
    ]
    # Run the command
    subprocess.run(command, check=True)

if __name__ == "__main__":
    input_video = "video.mp4"
    output_video = "converted_video.mp4"
    convert_video(input_video, output_video)