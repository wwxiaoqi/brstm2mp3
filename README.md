# brstm2mp3

把任天堂 **BRSTM**(Wii/3DS 游戏常用的 ADPCM 音乐容器)转成 **MP3 / FLAC / OGG / WAV** 的纯 Python 项目。

解码器是从零手写的:`BRSTM 容器解析 + DSP ADPCM 解码`全部用 Python + NumPy 实现,**不依赖任何现成的游戏音频库**;只有 MP3/FLAC/OGG 的最终编码交给 `ffmpeg`(WAV 输出完全不需要 ffmpeg)。

> 已用你提供的 `UNDERTALE - Undertale.brstm`(6:21,44100 Hz,立体声)验证:
> **解码结果与 ffmpeg 的 `adpcm_thp` 解码器逐采样完全一致(1670 万采样点,0 处差异)**。

---

## 1. 快速开始

```bash
# 依赖:Python 3.9+ / numpy;MP3 编码需要系统里有 ffmpeg
pip install numpy          # 或者 pip install -r requirements.txt

# 最简单:一个文件 -> 同目录下同名 .mp3
python -m brstm2mp3 "UNDERTALE - Undertale.brstm"

# 指定输出目录 / 码率
python -m brstm2mp3 song.brstm -o out -b 320k

# 整目录批量 + FLAC
python -m brstm2mp3 "D:\rips\ost" -o "D:\mp3" -f flac

# 先看文件信息(不转换)
python -m brstm2mp3 song.brstm --info
```

Windows 下也可以直接把 `.brstm` 文件**拖到 `convert.bat` 上**,同目录生成 MP3。

安装成命令(可选):

```bash
pip install .
brstm2mp3 song.brstm -f mp3 -b 320k
```

## 2. 命令行参数

| 参数 | 说明 |
| --- | --- |
| `inputs...` | BRSTM 文件、目录(自动找 `*.brstm`)或通配符 |
| `-o, --output DIR` | 输出目录,默认与源文件同目录 |
| `-f, --format` | `mp3`(默认) / `wav` / `flac` / `ogg` / `m4a` |
| `-b, --bitrate` | 有损格式码率,默认 `192k` |
| `--loops N` | 循环曲追加 **N 遍**循环段(0 = 原样播放一遍) |
| `--fade-in SEC` / `--fade-out SEC` | 淡入 / 淡出秒数 |
| `--trim-silence` | 去掉首尾数字静音 |
| `--decoder` | `auto`(默认,有 ffmpeg 时用它解码)/ `python`(本项目解码器)/ `ffmpeg` |
| `--info` | 只打印流信息 |
| `--overwrite` | 覆盖已存在的输出 |
| `-q, --quiet` | 不显示进度条 |

例子:做 OST 里常见的"循环两遍 + 8 秒淡出":

```bash
python -m brstm2mp3 "ost/*.brstm" -o mp3 --loops 2 --fade-out 8 -b 320k
```

## 3. 作为库使用

```python
from brstm2mp3 import BrstmFile, LoopOptions, looping, encoder

bf = BrstmFile.open("song.brstm")
print(bf.describe())                 # codec / 采样率 / 循环点 / 块布局

pcm = bf.decode()                    # numpy int16,形状 (采样数, 声道数)
pcm = looping.process(pcm, bf.stream.loop_flag, bf.stream.loop_start,
                      LoopOptions(loops=1, fade_out=5, sample_rate=bf.stream.sample_rate))
encoder.encode("song.mp3", pcm, bf.stream.sample_rate, fmt="mp3", bitrate="320k")
```

## 4. BRSTM 文件结构(本项目解析的部分)

BRSTM(`RSTM`)是**大端**的分块容器。头部 0x40 字节:

```
0x00 "RSTM"
0x04 u16 BOM(0xFEFF = 大端,0xFFFE = 小端)
0x06 u16 版本(0x0100)
0x08 u32 文件大小
0x0C u16 头部大小(0x40)
0x0E u16 段数量
0x10 { u32 偏移, u32 大小 } × 段数量      -> HEAD / ADPC / DATA
```

`HEAD` 段里放了三组引用(type + 偏移,偏移基准是 **HEAD 段起始 + 8**):
`流信息` / `音轨信息` / `声道信息`。

流信息(0x34 字节,解码需要的全在这):

| 偏移 | 类型 | 含义 |
| --- | --- | --- |
| 0x00 | u8 | 编码:0=PCM8, 1=PCM16, 2=DSP ADPCM |
| 0x01 | u8 | 是否循环 |
| 0x02 | u8 | 声道数 |
| 0x04 | u16 | 采样率 |
| 0x08 | u32 | 循环起点(采样数) |
| 0x0C | u32 | 总采样数 |
| 0x10 | u32 | 音频数据起始偏移(绝对) |
| 0x14 | u32 | 块数量 |
| 0x18 | u32 | 每块字节数 |
| 0x1C | u32 | 每块采样数 |
| 0x20 | u32 | 最后一块有效字节数 |
| 0x24 | u32 | 最后一块采样数 |
| 0x28 | u32 | 最后一块对齐后字节数(0x20 对齐) |
| 0x2C/0x30 | u32 | ADPC 每项采样数 / 字节数 |

声道信息段:`u32 声道数` + 每声道一个引用,指向的声道结构里第一个引用再指向
**32 字节 = 8 组 int16 预测系数**(Q11),也就是 ADPCM 解码表。

音频数据(`DATA` 段 + 0x20 处开始)按**块交错**存放:

```
ch0 block0 | ch1 block0 | ch0 block1 | ch1 block1 | ...   (最后一块只存一次,32 字节对齐)
```

## 5. DSP ADPCM 解码算法

每块按 8 字节一帧切开,一帧 14 个采样:

```
byte0       header:高 4 位 = 预测器编号 0..7,低 4 位 = 缩放 scale 0..12
byte1..7    14 个有符号 4bit nibble(高 nibble 在前)
```

重建公式(系数来自上面的 32 字节表):

```python
sample = clamp16( ( ((nib << scale) << 11) + c1*yn1 + c2*yn2 ) >> 11 )
```

其中 `yn1, yn2` 是前两个已解码采样。

**这里有个坑,也是最容易写错的地方**:`yn1/yn2` 的历史状态**在块与块之间是连续的**。
如果把每块当成独立单元、块首把历史清零,波形会在每个块边界(14336 采样 ≈ 0.32 s)
出现一个小台阶——听感上是每 0.3 秒一次的轻微"咔",而波形上看只是块首几十个采样
有一点偏差,很容易被当成"差不多对"而忽略。

本项目早期版本就踩了这个坑(和 ffmpeg 对比时相关系数 0.9996、只在块边界有小偏差),
改成**整条声道连续解码**后才做到逐采样完全一致:

```
ch0: IDENTICAL  max |diff| = 0   mismatched samples = 0
ch1: IDENTICAL  max |diff| = 0   mismatched samples = 0
```

此外还有两个细节:

* 最后一块名义上只用了 `last_block_used_bytes` 字节,但它是按 0x20 对齐存放的,
  且末尾几个有效采样要用到"补齐区"里的数据 —— 所以要按整块读(ffmpeg/vgmstream 也是这么做的),
  读不满就用文件末尾截断。
* 块交错的步长要算上最后一块的对齐长度,否则立体声第二条声道的最后一个块会算错偏移。

## 6. 项目结构

```
brstm2mp3/
├── brstm2mp3/
│   ├── __init__.py      包入口
│   ├── brstm.py         BRSTM 容器解析(BOM/段/HEAD/声道系数)
│   ├── pcm.py           DSP ADPCM / PCM16 / PCM8 解码(核心)
│   ├── builder.py       BRSTM 写入器(DSP ADPCM 编码器,用于自动化测试)
│   ├── looping.py       循环追加、淡入淡出、静音修剪
│   ├── encoder.py       WAV 写出 + ffmpeg 编码(mp3/flac/ogg/m4a)
│   └── cli.py           命令行
├── tests/test_core.py   单元测试(含与 ffmpeg 的逐采样对比)
├── verify.py            任意 brstm 与 ffmpeg 做逐采样对比的自检脚本
├── convert.bat          Windows 拖放即转
├── requirements.txt / pyproject.toml
└── out/                 示例输出(UNDERTALE - Undertale.mp3)
```

## 7. 自检 & 测试

```bash
python -m unittest -v tests.test_core   # 7 个测试,含 ffmpeg 逐采样对比
python verify.py "UNDERTALE - Undertale.brstm"
```

`verify.py` 会分别用本项目解码器和 ffmpeg 解码,然后逐采样比较:

```
== UNDERTALE - Undertale.brstm
   DSP ADPCM, 2 ch, 44100 Hz, 16825725 samples
   python decode 15.02s, ffmpeg decode 0.31s
   ch0: IDENTICAL  max |diff| = 0, mismatched samples = 0
   ch1: IDENTICAL  max |diff| = 0, mismatched samples = 0
```

## 8. 性能

| 步骤 | 6:21 的曲子 | 说明 |
| --- | --- | --- |
| Python 解码 | 15 ~ 25 s | 单线程、纯 CPython(算法天然串行,见上面的"跨块连续") |
| ffmpeg 解码 | ≈ 0.3 s | `--decoder ffmpeg` / 默认 `auto` 时使用 |
| MP3 320k 编码 | ≈ 6 s | ffmpeg libmp3lame |

想要极致速度就 `--decoder ffmpeg`;想验证/研究纯 Python 实现就用 `--decoder python`
(两者输出完全一致)。

## 9. 常见问题

* **提示找不到 ffmpeg**:装一个 ffmpeg 并加入 PATH,或设置环境变量 `FFMPEG` 指向
  `ffmpeg.exe` 的完整路径;只想拿到无损数据可以先 `-f wav`(不需要 ffmpeg)。
* **BCSTM / BFSTM / BRWAV**:那是 3DS/WiiU 的容器,格式不同,本项目不支持。
* **转换出来的曲子只有一遍**:BRSTM 里带循环标记时,默认按"完整播放一遍"输出,
  想要多遍用 `--loops N`。
* **文件名里带中文/空格**:支持,路径按 UTF-8 交给 Python 处理(命令行记得加引号)。

## 10. 许可与免责

代码 MIT。项目只做格式解析与转码,**不包含任何游戏音频资源**;
请仅对自己合法拥有的音频使用,转换出的音乐版权归原作者/发行方所有。
