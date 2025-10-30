# Step 1: 生成调色板（颜色优化）
ffmpeg -y -i 1_output.mp4 -vf "fps=5,scale=1440:-1:flags=lanczos,palettegen" palette.png

# Step 2: 使用调色板生成 gif
ffmpeg -i 1_output.mp4 -i palette.png -filter_complex "fps=5,scale=1440:-1:flags=lanczos[x];[x][1:v]paletteuse" output.gif
