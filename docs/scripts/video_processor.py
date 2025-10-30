import subprocess
import shlex

def build_segments(trims, speedups):
    """把 trims 和 speedups 合并成连续段 [(start,end,speed)]"""
    segments = []
    for t_start, t_end in trims:
        cur = t_start
        for s_start, s_end, factor in sorted(speedups):
            if s_end <= t_start or s_start >= t_end:
                continue
            if cur < s_start:
                segments.append((cur, min(s_start, t_end), 1.0))
            segments.append((max(s_start, t_start), min(s_end, t_end), factor))
            cur = s_end
        if cur < t_end:
            segments.append((cur, t_end, 1.0))
    return segments


def _escape_drawtext_text(s: str) -> str:
    # ffmpeg drawtext 使用单引号包裹时，需要把单引号转成 \' （或者用双引号并转义）
    # 这里简单把单引号替换为 \'
    return s.replace("'", r"\'")


def process_video(input_path, output_path, trims=None, speedups=None, texts=None,
                  crf=25, preset="medium",
                  font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    trims = trims or []
    speedups = speedups or []
    texts = texts or []

    segments = build_segments(trims, speedups)
    if not segments:
        raise ValueError("no segments built — check trims/speedups")

    filter_parts = []
    concat_labels = []

    # 每段独立处理
    for i, (start, end, factor) in enumerate(segments):
        vtrim = f"[v{i}_trim]"
        vspd = f"[v{i}_spd]" if factor != 1.0 else vtrim

        # trim -> reset timestamps to 0..(end-start)
        filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS{vtrim}")

        # 如果有加速：把时间轴缩放（播放更快）。注意 setpts=PTS/factor 会把流上的时间缩短为 (原相对时间)/factor
        if factor != 1.0:
            filter_parts.append(f"{vtrim}setpts=PTS/{factor}{vspd}")

        # 在该段上叠加所有与该段有重叠的文本，注意 drawtext 的 enable 使用的是“段内播放时间”
        vtext = vspd
        for j, t in enumerate(texts):
            # 文本在原始时间轴上是否与本段有重叠
            if t['end'] <= start or t['start'] >= end:
                continue

            # 计算本段内显示时间（段内播放时间）
            # 原始相对时间（相对于 segment.start）
            rel_start = max(t['start'], start) - start
            rel_end   = min(t['end'], end) - start

            # 如果段被加速，流上的时间会缩短：t_stream = rel / factor
            if factor != 1.0:
                ts = rel_start / factor
                te = rel_end / factor
            else:
                ts = rel_start
                te = rel_end

            # 保护性检查：避免负值或 te<=ts
            if te <= 0 or ts >= (end - start) / (factor if factor != 1.0 else 1.0):
                # 虽然上面已过滤重叠，但加速后可能导致时间在流上超范围，跳过
                continue

            label = f"[v{i}_txt{j}]"
            txt = _escape_drawtext_text(t['text'])
            # drawtext 中 text 的换行 \n 会被 ffmpeg 识别
            filter_parts.append(
                f"{vtext}drawtext=text='{txt}':"
                f"x={t['x']}:y={t['y']}:fontsize={t['fontsize']}:fontcolor={t['color']}:"
                f"fontfile='{font_path}':"
                f"enable='between(t,{ts},{te})'{label}"
            )
            vtext = label

        concat_labels.append(vtext)

    # 拼接所有段（注意 n 要等于段数）
    filter_parts.append(f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=1:a=0[vout]")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        output_path
    ]

    # 打印并执行（如果命令太长，打印换行更可读）
    print("ffmpeg command:", " \\\n  ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)




# 示例调用
if __name__ == "__main__":
    video = 4
    if video == 1:
        # 基本功能演示
        video_name = "1"
        trims = [(0., 242.)]
        speedups = [
            (5., 30., 4.0),
            (80., 125., 4.0),
            (130., 235., 8.0)
        ]
        texts = [
            {"text": "IsaacSim仿真环境\n宇树Go2W\nRL行走运控\n开放场景演示\n通过ROS2同智能体后台通信", "start": 0, "end": 240, "x": 300, "y": 200, "fontsize": 64, "color": "yellow"},
            {"text": "后台自动建图\n拖动查看地图", "start": 30, "end": 240, "x": 3800, "y": 200, "fontsize": 64, "color": "yellow"},
            {"text": "启动等待中\nx4倍速播放中", "start": 5, "end": 30, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "手操移动中", "start": 35, "end": 75, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "自主探索中\nx4倍速播放中", "start": 80, "end": 125, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "手操移动中\nx8播放中", "start": 130, "end": 235, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "保存地图中", "start": 235, "end": 242, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
        ]
    elif video == 2:
        # 街道开放指令演示
        video_name = "2"
        trims = [(0, 195)]
        speedups = [
            (0., 20., 4.0),    # 启动等待中
            (30., 55., 4.0),    # 输入指令中
            (65., 95., 4.0),    # 前往车辆处
            (110., 155., 4.0)   # 前往交通护栏
        ]
        texts = [
            {"text": "IsaacSim仿真环境\n宇树Go2W\nRL行走运控\n开放场景演示\n通过ROS2同智能体后台通信", "start": 0, "end": 240, "x": 300, "y": 200, "fontsize": 64, "color": "yellow"},
            {"text": "智能体生成底层指令自主移动中", "start": 65, "end": 240, "x": 1200, "y": 700, "fontsize": 64, "color": "yellow"},
            {"text": "后台自动建图\n拖动查看地图", "start": 30, "end": 240, "x": 3800, "y": 200, "fontsize": 64, "color": "yellow"},
            {"text": "载入预建地图中", "start": 5, "end": 25, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "输入指令中\nx4倍速播放中\n", "start": 30, "end": 55, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "当前指令：请前往拍摄交通路障,并在车辆处标记", "start": 30, "end": 165, "x": 2800, "y": 400, "fontsize": 72, "color": "yellow"},
            {"text": "前往车辆处\nx4倍速播放中", "start": 65, "end": 95, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "前往交通护栏\nx4倍速播放中", "start": 110, "end": 155, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "生成行为树", "start": 56, "end": 64, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "目标标记插旗", "start": 96, "end": 104, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "查看拍摄图片", "start": 156, "end": 166, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "继续发送指令作业中...", "start": 170, "end": 195, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
        ]
    elif video == 3:
        # 仓库自主探索建图
        video_name = "3"
        trims = [(0, 190)]
        speedups = [
            (0., 175., 4.0),   
        ]
        texts = [
            {"text": "IsaacSim仿真环境\n宇树Go2W\nRL行走运控\n开放场景演示\n通过ROS2同智能体后台通信", "start": 0, "end": 190, "x": 300, "y": 200, "fontsize": 64, "color": "yellow"},
            {"text": "后台自动建图\n拖动查看地图", "start": 0, "end": 190, "x": 3800, "y": 200, "fontsize": 64, "color": "yellow"},
            {"text": "智能体规划自主探索中\nx4倍速播放中", "start": 6, "end": 170, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "查看并保存地图", "start": 175, "end": 185, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
        ]
    elif video == 4:
        # 仓库开放指令演示
        video_name = "4"
        trims = [(0, 360)]
        speedups = [
            (0., 155., 10.0),
            (160., 360., 4.0),
        ]
        texts = [
            {"text": "IsaacSim仿真环境\n宇树Go2W\nRL行走运控\n开放场景演示\n通过ROS2同智能体后台通信", "start": 0, "end": 360, "x": 300, "y": 200, "fontsize": 64, "color": "yellow"},
            {"text": "后台自动建图\n拖动查看地图", "start": 0, "end": 360, "x": 3800, "y": 200, "fontsize": 64, "color": "yellow"},
            {"text": "继续自主探索更新旧地图", "start": 0, "end": 155, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
            {"text": "指令：请前往拍摄任一卡通人物牌", "start": 165, "end": 220, "x": 2800, "y": 400, "fontsize": 72, "color": "yellow"},
            {"text": "指令：前往卡牌盒处标记!", "start": 230, "end": 270, "x": 2800, "y": 400, "fontsize": 72, "color": "yellow"},
            {"text": "指令：再去梯子处拍照", "start": 280, "end": 350, "x": 2800, "y": 400, "fontsize": 72, "color": "yellow"},
            {"text": "保存更新后的地图", "start": 352, "end": 360, "x": 2800, "y": 200, "fontsize": 72, "color": "yellow"},
        ]

    process_video(
        input_path=f"/media/lenovo/1/Videos/{video_name}.mp4",
        output_path=f"/home/lenovo/Videos/{video_name}_output.mp4",
        trims=trims,
        speedups=speedups,
        texts=texts,
        crf=23,
        preset="medium"
    )
