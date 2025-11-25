# Step 0: 加速加fps
ffmpeg -hwaccel cuda -i input.mp4 \
  -vf "setpts=PTS/2,fps=30" \
  -af "atempo=2.0" \
  -t 00:05:45 \
  -c:v h264_nvenc -preset fast \
  output.mp4

# Step 1: 生成调色板（颜色优化）
ffmpeg -y -i 1_output.mp4 -vf "fps=5,scale=1440:-1:flags=lanczos,palettegen" palette.png

# Step 2: 使用调色板生成 gif
ffmpeg -i 1_output.mp4 -i palette.png -filter_complex "fps=5,scale=1440:-1:flags=lanczos[x];[x][1:v]paletteuse" output.gif
