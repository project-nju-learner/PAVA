#!/usr/bin/env python3
"""冻结 597 题所需的 465 个视频：公开源下载（Ubuntu，Python 3.9+，无 pip 依赖）。

只需要把本文件交给对方；465 个视频名单已经内置，不需要原服务器权限。
运行命令（一行）：python3 download_videos.py /data/rebuttal597
仅看计划（一行，不联网）：python3 download_videos.py --plan
后台运行（一行）：tmux new-session -s videos597 'python3 download_videos.py /data/rebuttal597'
进入 tmux 后，如提示请输入自己的 HF token；Ctrl+B 再 D 可离开会话。

开始前：
1. LongVideoBench 官方源要求登录并同意条款：
   https://huggingface.co/datasets/longvideobench/LongVideoBench
   用自己的 Hugging Face 账号取得访问权。本文件不包含任何账号或密钥。
   支持 HF_TOKEN 环境变量、现有 hf auth login 凭据，或交互式隐藏输入；不写入 token。
2. 准备至少 250 GB 可用磁盘。采用常规完整分卷下载，网络流量约 600 GB，
   不是只传 122.1 GB。每卷只提取选中视频，成功后删除本程序的缓存分卷。
   LongVideoBench 需 31 个 tar 分卷，合计约 161.6 GB；不再生成一份合并 tar。

固定范围：LVBench 180 题/87 视频；LongVideoBench 150/136；
Video-MME Long 140/121；MLVU 127/121，共 597 题/465 视频。没有重新抽样。
MLVU 使用 sy1998/MLVU_dev 公开开发集镜像。
公开视频约 115.1 GB。之前 122.1 GB 是现有本地文件大小；其中 5 个 LVBench
文件有本地重封装/转码版本。本脚本保留公开源原视频，不保证与本地版本逐字节相同。
这里只下载视频，不含模型权重、ASR、OpenCLIP 索引、初始帧缓存或推理运行环境。

中断后重新运行相同命令：压缩分卷断点续传，完整视频跳过；不自动更换视频。
文件保存在 <目标目录>/<数据集>/videos/<官方文件名>。
全部完成后生成 video_manifest.json，保留视频 ID、相对路径和公开归档来源。
"""

import argparse
from collections import Counter, defaultdict
import getpass
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


SOURCES = {
    "LVBench": "lmms-eval/LVBench",
    "LongVideoBench": "longvideobench/LongVideoBench",
    "Video-MME": "lmms-eval/Video-MME",
    "MLVU": "sy1998/MLVU_dev",
}

# Frozen public-source filenames and archive members; no questions or answer labels.
PLAN = json.loads(r"""{"question_count":597,"seed":42,"catalog":{"LVBench":[{"path":"video_chunks/videos_chunk_001.zip","size":3418973560},{"path":"video_chunks/videos_chunk_002.zip","size":2984952861},{"path":"video_chunks/videos_chunk_003.zip","size":5036119592},{"path":"video_chunks/videos_chunk_004.zip","size":4138587819},{"path":"video_chunks/videos_chunk_005.zip","size":4197860091},{"path":"video_chunks/videos_chunk_006.zip","size":4613116799},{"path":"video_chunks/videos_chunk_007.zip","size":4700749936},{"path":"video_chunks/videos_chunk_008.zip","size":4892078458},{"path":"video_chunks/videos_chunk_009.zip","size":4856143323},{"path":"video_chunks/videos_chunk_010.zip","size":5083286141},{"path":"video_chunks/videos_chunk_011.zip","size":5250736434},{"path":"video_chunks/videos_chunk_012.zip","size":5284578644},{"path":"video_chunks/videos_chunk_013.zip","size":5288068238},{"path":"video_chunks/videos_chunk_014.zip","size":2081649135}],"MLVU":[{"path":"video_part_1.zip","size":36626152774},{"path":"video_part_2.zip","size":34311867313},{"path":"video_part_3.zip","size":41427664931},{"path":"video_part_4.zip","size":43162717558},{"path":"video_part_5.zip","size":44702247576},{"path":"video_part_6.zip","size":29624788310},{"path":"video_part_7.zip","size":33061239060},{"path":"video_part_8.zip","size":22147421187}],"LongVideoBench":[{"path":"videos.tar.part.aa","size":5242880000},{"path":"videos.tar.part.ab","size":5242880000},{"path":"videos.tar.part.ac","size":5242880000},{"path":"videos.tar.part.ad","size":5242880000},{"path":"videos.tar.part.ae","size":5242880000},{"path":"videos.tar.part.af","size":5242880000},{"path":"videos.tar.part.ag","size":5242880000},{"path":"videos.tar.part.ah","size":5242880000},{"path":"videos.tar.part.ai","size":5242880000},{"path":"videos.tar.part.aj","size":5242880000},{"path":"videos.tar.part.ak","size":5242880000},{"path":"videos.tar.part.al","size":5242880000},{"path":"videos.tar.part.am","size":5242880000},{"path":"videos.tar.part.an","size":5242880000},{"path":"videos.tar.part.ao","size":5242880000},{"path":"videos.tar.part.ap","size":5242880000},{"path":"videos.tar.part.aq","size":5242880000},{"path":"videos.tar.part.ar","size":5242880000},{"path":"videos.tar.part.as","size":5242880000},{"path":"videos.tar.part.at","size":5242880000},{"path":"videos.tar.part.au","size":5242880000},{"path":"videos.tar.part.av","size":5242880000},{"path":"videos.tar.part.aw","size":5242880000},{"path":"videos.tar.part.ax","size":5242880000},{"path":"videos.tar.part.ay","size":5242880000},{"path":"videos.tar.part.az","size":5242880000},{"path":"videos.tar.part.ba","size":5242880000},{"path":"videos.tar.part.bb","size":5242880000},{"path":"videos.tar.part.bc","size":5242880000},{"path":"videos.tar.part.bd","size":5242880000},{"path":"videos.tar.part.be","size":4277780480}]},"rows":[
{"dataset":"LVBench","video_id":"rk24OUu_kJQ","filename":"rk24OUu_kJQ.mp4","archive":"video_chunks/videos_chunk_009.zip","member":"rk24OUu_kJQ.mp4","bytes":633595340},
{"dataset":"LVBench","video_id":"8QdE--Y-x7U","filename":"8QdE--Y-x7U.mp4","archive":"video_chunks/videos_chunk_006.zip","member":"8QdE--Y-x7U.mp4","bytes":1116790035},
{"dataset":"LVBench","video_id":"1FsiZgGZU70","filename":"1FsiZgGZU70.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"1FsiZgGZU70.mp4","bytes":207190230},
{"dataset":"LVBench","video_id":"wgBlACG927Y","filename":"wgBlACG927Y.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"wgBlACG927Y.mp4","bytes":613030025},
{"dataset":"LVBench","video_id":"QB7FoIpx8os","filename":"QB7FoIpx8os.mp4","archive":"video_chunks/videos_chunk_009.zip","member":"QB7FoIpx8os.mp4","bytes":693278185},
{"dataset":"LVBench","video_id":"KktLi3UifPY","filename":"KktLi3UifPY.mp4","archive":"video_chunks/videos_chunk_008.zip","member":"KktLi3UifPY.mp4","bytes":781466062},
{"dataset":"LVBench","video_id":"JlrzSvCsIjE","filename":"JlrzSvCsIjE.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"JlrzSvCsIjE.mp4","bytes":326845222},
{"dataset":"LVBench","video_id":"CgvJqGxzRfE","filename":"CgvJqGxzRfE.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"CgvJqGxzRfE.mp4","bytes":453468825},
{"dataset":"LVBench","video_id":"8NHmfpkxTSw","filename":"8NHmfpkxTSw.mp4","archive":"video_chunks/videos_chunk_007.zip","member":"8NHmfpkxTSw.mp4","bytes":985803601},
{"dataset":"LVBench","video_id":"tH_5YbklevQ","filename":"tH_5YbklevQ.mp4","archive":"video_chunks/videos_chunk_007.zip","member":"tH_5YbklevQ.mp4","bytes":929007427},
{"dataset":"LVBench","video_id":"jp2M1hIEtsk","filename":"jp2M1hIEtsk.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"jp2M1hIEtsk.mp4","bytes":156737606},
{"dataset":"LVBench","video_id":"7HjNIPIgUg4","filename":"7HjNIPIgUg4.mp4","archive":"video_chunks/videos_chunk_006.zip","member":"7HjNIPIgUg4.mp4","bytes":1169981898},
{"dataset":"LVBench","video_id":"pXD3txG2bVQ","filename":"pXD3txG2bVQ.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"pXD3txG2bVQ.mp4","bytes":249951440},
{"dataset":"LVBench","video_id":"Z4HGQL_McDQ","filename":"Z4HGQL_McDQ.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"Z4HGQL_McDQ.mp4","bytes":239047290},
{"dataset":"LVBench","video_id":"20lTg3yUrO4","filename":"20lTg3yUrO4.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"20lTg3yUrO4.mp4","bytes":531337678},
{"dataset":"LVBench","video_id":"81SbCR6p3Z0","filename":"81SbCR6p3Z0.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"81SbCR6p3Z0.mp4","bytes":125457386},
{"dataset":"LVBench","video_id":"KbahC-QCKU8","filename":"KbahC-QCKU8.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"KbahC-QCKU8.mp4","bytes":618839868},
{"dataset":"LVBench","video_id":"gbDR39yIs3Y","filename":"gbDR39yIs3Y.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"gbDR39yIs3Y.mp4","bytes":165149669},
{"dataset":"LVBench","video_id":"pe_LddfHAUU","filename":"pe_LddfHAUU.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"pe_LddfHAUU.mp4","bytes":437174334},
{"dataset":"LVBench","video_id":"lDlA7cfNk8A","filename":"lDlA7cfNk8A.mp4","archive":"video_chunks/videos_chunk_009.zip","member":"lDlA7cfNk8A.mp4","bytes":702951358},
{"dataset":"LVBench","video_id":"IsdbCjlZ5cQ","filename":"IsdbCjlZ5cQ.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"IsdbCjlZ5cQ.mp4","bytes":371800111},
{"dataset":"LVBench","video_id":"vZV2WCKMsKs","filename":"vZV2WCKMsKs.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"vZV2WCKMsKs.mp4","bytes":617211792},
{"dataset":"LVBench","video_id":"rp4NKWb7dXk","filename":"rp4NKWb7dXk.mp4","archive":"video_chunks/videos_chunk_008.zip","member":"rp4NKWb7dXk.mp4","bytes":850377960},
{"dataset":"LVBench","video_id":"vHlSoxg8WHo","filename":"vHlSoxg8WHo.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"vHlSoxg8WHo.mp4","bytes":228274535},
{"dataset":"LVBench","video_id":"_T2Avd3tFHc","filename":"_T2Avd3tFHc.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"_T2Avd3tFHc.mp4","bytes":271615396},
{"dataset":"LVBench","video_id":"oZEVgDXJwCc","filename":"oZEVgDXJwCc.mp4","archive":"video_chunks/videos_chunk_004.zip","member":"oZEVgDXJwCc.mp4","bytes":2370014302},
{"dataset":"LVBench","video_id":"-hgaSElC3wU","filename":"-hgaSElC3wU.mp4","archive":"video_chunks/videos_chunk_005.zip","member":"-hgaSElC3wU.mp4","bytes":1563462994},
{"dataset":"LVBench","video_id":"EwskdNETNx8","filename":"EwskdNETNx8.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"EwskdNETNx8.mp4","bytes":340045941},
{"dataset":"LVBench","video_id":"uusf1qG_uZ4","filename":"uusf1qG_uZ4.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"uusf1qG_uZ4.mp4","bytes":402358486},
{"dataset":"LVBench","video_id":"T1yhBv1ytzw","filename":"T1yhBv1ytzw.mp4","archive":"video_chunks/videos_chunk_003.zip","member":"T1yhBv1ytzw.mp4","bytes":2489758770},
{"dataset":"LVBench","video_id":"EpMLAQbSYAw","filename":"EpMLAQbSYAw.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"EpMLAQbSYAw.mp4","bytes":411594832},
{"dataset":"LVBench","video_id":"JTa_Ue2MSwc","filename":"JTa_Ue2MSwc.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"JTa_Ue2MSwc.mp4","bytes":191834878},
{"dataset":"LVBench","video_id":"Va_9Q6ekm60","filename":"Va_9Q6ekm60.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"Va_9Q6ekm60.mp4","bytes":292233350},
{"dataset":"LVBench","video_id":"TiQBTesZUJQ","filename":"TiQBTesZUJQ.mp4","archive":"video_chunks/videos_chunk_005.zip","member":"TiQBTesZUJQ.mp4","bytes":1209865694},
{"dataset":"LVBench","video_id":"TJR1oYDDTwg","filename":"TJR1oYDDTwg.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"TJR1oYDDTwg.mp4","bytes":552049066},
{"dataset":"LVBench","video_id":"NzCO0G8AGLU","filename":"NzCO0G8AGLU.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"NzCO0G8AGLU.mp4","bytes":159028596},
{"dataset":"LVBench","video_id":"2LH3JCGkEBU","filename":"2LH3JCGkEBU.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"2LH3JCGkEBU.mp4","bytes":479367627},
{"dataset":"LVBench","video_id":"wNCPgIVz15c","filename":"wNCPgIVz15c.mp4","archive":"video_chunks/videos_chunk_009.zip","member":"wNCPgIVz15c.mp4","bytes":753528099},
{"dataset":"LVBench","video_id":"_zrgbA3FMVE","filename":"_zrgbA3FMVE.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"_zrgbA3FMVE.mp4","bytes":380795800},
{"dataset":"LVBench","video_id":"ihfjEFGdZdc","filename":"ihfjEFGdZdc.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"ihfjEFGdZdc.mp4","bytes":278518292},
{"dataset":"LVBench","video_id":"AeEYQ62t8hA","filename":"AeEYQ62t8hA.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"AeEYQ62t8hA.mp4","bytes":162006345},
{"dataset":"LVBench","video_id":"5dZ_lvDgevk","filename":"5dZ_lvDgevk.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"5dZ_lvDgevk.mp4","bytes":415141407},
{"dataset":"LVBench","video_id":"QgWRyDV9Ozs","filename":"QgWRyDV9Ozs.mp4","archive":"video_chunks/videos_chunk_009.zip","member":"QgWRyDV9Ozs.mp4","bytes":699429634},
{"dataset":"LVBench","video_id":"qYMnM5blZIE","filename":"qYMnM5blZIE.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"qYMnM5blZIE.mp4","bytes":419151042},
{"dataset":"LVBench","video_id":"qAIRFyR6NyQ","filename":"qAIRFyR6NyQ.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"qAIRFyR6NyQ.mp4","bytes":232011684},
{"dataset":"LVBench","video_id":"o-gLbgpzCc8","filename":"o-gLbgpzCc8.mp4","archive":"video_chunks/videos_chunk_008.zip","member":"o-gLbgpzCc8.mp4","bytes":845073642},
{"dataset":"LVBench","video_id":"HfEVEGf1A8Q","filename":"HfEVEGf1A8Q.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"HfEVEGf1A8Q.mp4","bytes":136604069},
{"dataset":"LVBench","video_id":"3_upA09AntU","filename":"3_upA09AntU.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"3_upA09AntU.mp4","bytes":481143884},
{"dataset":"LVBench","video_id":"t-RtDI2RWQs","filename":"t-RtDI2RWQs.mp4","archive":"video_chunks/videos_chunk_009.zip","member":"t-RtDI2RWQs.mp4","bytes":686683593},
{"dataset":"LVBench","video_id":"KLIa2UaE2KE","filename":"KLIa2UaE2KE.mp4","archive":"video_chunks/videos_chunk_002.zip","member":"KLIa2UaE2KE.mp4","bytes":2985453701},
{"dataset":"LVBench","video_id":"QWXlvx1GoTY","filename":"QWXlvx1GoTY.mp4","archive":"video_chunks/videos_chunk_008.zip","member":"QWXlvx1GoTY.mp4","bytes":803863608},
{"dataset":"LVBench","video_id":"rSE2YPcv89U","filename":"rSE2YPcv89U.mp4","archive":"video_chunks/videos_chunk_003.zip","member":"rSE2YPcv89U.mp4","bytes":2547204804},
{"dataset":"LVBench","video_id":"USfvmoalqsw","filename":"USfvmoalqsw.mp4","archive":"video_chunks/videos_chunk_006.zip","member":"USfvmoalqsw.mp4","bytes":1189466178},
{"dataset":"LVBench","video_id":"TZ0j6kr4ZJ0","filename":"TZ0j6kr4ZJ0.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"TZ0j6kr4ZJ0.mp4","bytes":334599048},
{"dataset":"LVBench","video_id":"O14bbpvy2x0","filename":"O14bbpvy2x0.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"O14bbpvy2x0.mp4","bytes":368139739},
{"dataset":"LVBench","video_id":"tKIFQI9cH2c","filename":"tKIFQI9cH2c.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"tKIFQI9cH2c.mp4","bytes":458890710},
{"dataset":"LVBench","video_id":"q01CUy_gwdU","filename":"q01CUy_gwdU.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"q01CUy_gwdU.mp4","bytes":430249368},
{"dataset":"LVBench","video_id":"FaV0tIaWWEg","filename":"FaV0tIaWWEg.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"FaV0tIaWWEg.mp4","bytes":194360418},
{"dataset":"LVBench","video_id":"idZkam9zqAs","filename":"idZkam9zqAs.mp4","archive":"video_chunks/videos_chunk_005.zip","member":"idZkam9zqAs.mp4","bytes":1428270467},
{"dataset":"LVBench","video_id":"uW9mcG0rdLY","filename":"uW9mcG0rdLY.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"uW9mcG0rdLY.mp4","bytes":222267257},
{"dataset":"LVBench","video_id":"k2FIFQIYBvA","filename":"k2FIFQIYBvA.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"k2FIFQIYBvA.mp4","bytes":141472445},
{"dataset":"LVBench","video_id":"S8vPx-u9p_A","filename":"S8vPx-u9p_A.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"S8vPx-u9p_A.mp4","bytes":221991741},
{"dataset":"LVBench","video_id":"2sriHX3PbXw","filename":"2sriHX3PbXw.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"2sriHX3PbXw.mp4","bytes":413464073},
{"dataset":"LVBench","video_id":"RjDrZkBwzho","filename":"RjDrZkBwzho.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"RjDrZkBwzho.mp4","bytes":570759623},
{"dataset":"LVBench","video_id":"XNtNNplAwiI","filename":"XNtNNplAwiI.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"XNtNNplAwiI.mp4","bytes":285161357},
{"dataset":"LVBench","video_id":"moALQl818ZY","filename":"moALQl818ZY.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"moALQl818ZY.mp4","bytes":361638402},
{"dataset":"LVBench","video_id":"sk00epALZps","filename":"sk00epALZps.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"sk00epALZps.mp4","bytes":301346808},
{"dataset":"LVBench","video_id":"gXnhqF0TqqI","filename":"gXnhqF0TqqI.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"gXnhqF0TqqI.mp4","bytes":276007868},
{"dataset":"LVBench","video_id":"W-BnDvXXfOs","filename":"W-BnDvXXfOs.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"W-BnDvXXfOs.mp4","bytes":422414344},
{"dataset":"LVBench","video_id":"LLSJrEgOOtw","filename":"LLSJrEgOOtw.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"LLSJrEgOOtw.mp4","bytes":376298584},
{"dataset":"LVBench","video_id":"xECIRjlxM3U","filename":"xECIRjlxM3U.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"xECIRjlxM3U.mp4","bytes":226541010},
{"dataset":"LVBench","video_id":"f8DKD78BrQA","filename":"f8DKD78BrQA.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"f8DKD78BrQA.mp4","bytes":605375199},
{"dataset":"LVBench","video_id":"xi6r3hZe5Tg","filename":"xi6r3hZe5Tg.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"xi6r3hZe5Tg.mp4","bytes":332195021},
{"dataset":"LVBench","video_id":"Cm73ma6Ibcs","filename":"Cm73ma6Ibcs.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"Cm73ma6Ibcs.mp4","bytes":205087832},
{"dataset":"LVBench","video_id":"2zkJFv-ro4A","filename":"2zkJFv-ro4A.mp4","archive":"video_chunks/videos_chunk_010.zip","member":"2zkJFv-ro4A.mp4","bytes":506006966},
{"dataset":"LVBench","video_id":"Vk_Af0htZGU","filename":"Vk_Af0htZGU.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"Vk_Af0htZGU.mp4","bytes":107456604},
{"dataset":"LVBench","video_id":"aJI8XTa_DII","filename":"aJI8XTa_DII.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"aJI8XTa_DII.mp4","bytes":220116657},
{"dataset":"LVBench","video_id":"Mcggugol2ts","filename":"Mcggugol2ts.mp4","archive":"video_chunks/videos_chunk_013.zip","member":"Mcggugol2ts.mp4","bytes":202306212},
{"dataset":"LVBench","video_id":"16Z-XQh9jhk","filename":"16Z-XQh9jhk.mp4","archive":"video_chunks/videos_chunk_009.zip","member":"16Z-XQh9jhk.mp4","bytes":705697240},
{"dataset":"LVBench","video_id":"vaL_vSdZKZo","filename":"vaL_vSdZKZo.mp4","archive":"video_chunks/videos_chunk_014.zip","member":"vaL_vSdZKZo.mp4","bytes":137518295},
{"dataset":"LVBench","video_id":"9-gOCOu_KGU","filename":"9-gOCOu_KGU.mp4","archive":"video_chunks/videos_chunk_006.zip","member":"9-gOCOu_KGU.mp4","bytes":1142443424},
{"dataset":"LVBench","video_id":"Z86xysw5Ncc","filename":"Z86xysw5Ncc.mp4","archive":"video_chunks/videos_chunk_007.zip","member":"Z86xysw5Ncc.mp4","bytes":942840862},
{"dataset":"LVBench","video_id":"-WnyRMZqV1U","filename":"-WnyRMZqV1U.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"-WnyRMZqV1U.mp4","bytes":362676630},
{"dataset":"LVBench","video_id":"GcRKREorGSc","filename":"GcRKREorGSc.mp4","archive":"video_chunks/videos_chunk_011.zip","member":"GcRKREorGSc.mp4","bytes":455321827},
{"dataset":"LVBench","video_id":"RCAqKnvu_P0","filename":"RCAqKnvu_P0.mp4","archive":"video_chunks/videos_chunk_012.zip","member":"RCAqKnvu_P0.mp4","bytes":385935220},
{"dataset":"LVBench","video_id":"VTCDQYYKA9o","filename":"VTCDQYYKA9o.mp4","archive":"video_chunks/videos_chunk_007.zip","member":"VTCDQYYKA9o.mp4","bytes":951290302},
{"dataset":"LVBench","video_id":"evYm0cELO3I","filename":"evYm0cELO3I.mp4","archive":"video_chunks/videos_chunk_008.zip","member":"evYm0cELO3I.mp4","bytes":814239701},
{"dataset":"LongVideoBench","video_id":"lvb_ysRFFN5nzqE","filename":"ysRFFN5nzqE.mp4","archive":"videos.tar","member":"videos/ysRFFN5nzqE.mp4","bytes":86659028},
{"dataset":"LongVideoBench","video_id":"lvb_@jonijawne-7214559428471770374","filename":"7214559428471770374.mp4","archive":"videos.tar","member":"videos/7214559428471770374.mp4","bytes":3799226},
{"dataset":"LongVideoBench","video_id":"lvb_1pxrIj9Xyps","filename":"1pxrIj9Xyps.mp4","archive":"videos.tar","member":"videos/1pxrIj9Xyps.mp4","bytes":118569524},
{"dataset":"LongVideoBench","video_id":"lvb_NAJOZTNkhlI","filename":"NAJOZTNkhlI.mp4","archive":"videos.tar","member":"videos/NAJOZTNkhlI.mp4","bytes":66463704},
{"dataset":"LongVideoBench","video_id":"lvb_KTY9bogonyw","filename":"KTY9bogonyw.mp4","archive":"videos.tar","member":"videos/KTY9bogonyw.mp4","bytes":91765165},
{"dataset":"LongVideoBench","video_id":"lvb_IGmuaY1jB1w","filename":"IGmuaY1jB1w.mp4","archive":"videos.tar","member":"videos/IGmuaY1jB1w.mp4","bytes":17269443},
{"dataset":"LongVideoBench","video_id":"lvb_@movie.explained6-7276598470151032066","filename":"7276598470151032066.mp4","archive":"videos.tar","member":"videos/7276598470151032066.mp4","bytes":2341544},
{"dataset":"LongVideoBench","video_id":"lvb_@healthfood-7135100078221430058","filename":"7135100078221430058.mp4","archive":"videos.tar","member":"videos/7135100078221430058.mp4","bytes":2787330},
{"dataset":"LongVideoBench","video_id":"lvb_mld0TnA2jEs","filename":"mld0TnA2jEs.mp4","archive":"videos.tar","member":"videos/mld0TnA2jEs.mp4","bytes":12102981},
{"dataset":"LongVideoBench","video_id":"lvb_97nEBjiQI1M","filename":"97nEBjiQI1M.mp4","archive":"videos.tar","member":"videos/97nEBjiQI1M.mp4","bytes":34790560},
{"dataset":"LongVideoBench","video_id":"lvb_sWfcgeDth_w","filename":"sWfcgeDth_w.mp4","archive":"videos.tar","member":"videos/sWfcgeDth_w.mp4","bytes":4017062},
{"dataset":"LongVideoBench","video_id":"lvb_acAWfzV__XI","filename":"acAWfzV__XI.mp4","archive":"videos.tar","member":"videos/acAWfzV__XI.mp4","bytes":29588899},
{"dataset":"LongVideoBench","video_id":"lvb_2ekjGl8yWZk","filename":"2ekjGl8yWZk.mp4","archive":"videos.tar","member":"videos/2ekjGl8yWZk.mp4","bytes":31537205},
{"dataset":"LongVideoBench","video_id":"lvb_2edlqFUTDVc","filename":"2edlqFUTDVc.mp4","archive":"videos.tar","member":"videos/2edlqFUTDVc.mp4","bytes":46819897},
{"dataset":"LongVideoBench","video_id":"lvb_9dSkvxS2EB0","filename":"9dSkvxS2EB0.mp4","archive":"videos.tar","member":"videos/9dSkvxS2EB0.mp4","bytes":66073103},
{"dataset":"LongVideoBench","video_id":"lvb_HWyDOQrYtCk","filename":"HWyDOQrYtCk.mp4","archive":"videos.tar","member":"videos/HWyDOQrYtCk.mp4","bytes":6022038},
{"dataset":"LongVideoBench","video_id":"lvb_JLBsG65WoVU","filename":"JLBsG65WoVU.mp4","archive":"videos.tar","member":"videos/JLBsG65WoVU.mp4","bytes":23637457},
{"dataset":"LongVideoBench","video_id":"lvb_jJGbXCCU5yc","filename":"jJGbXCCU5yc.mp4","archive":"videos.tar","member":"videos/jJGbXCCU5yc.mp4","bytes":24340306},
{"dataset":"LongVideoBench","video_id":"lvb_tdm72-vYxTs","filename":"tdm72-vYxTs.mp4","archive":"videos.tar","member":"videos/tdm72-vYxTs.mp4","bytes":274344642},
{"dataset":"LongVideoBench","video_id":"lvb_22iOyzE8Ec0","filename":"22iOyzE8Ec0.mp4","archive":"videos.tar","member":"videos/22iOyzE8Ec0.mp4","bytes":27693710},
{"dataset":"LongVideoBench","video_id":"lvb_pFtKaT3GF9I","filename":"pFtKaT3GF9I.mp4","archive":"videos.tar","member":"videos/pFtKaT3GF9I.mp4","bytes":44567722},
{"dataset":"LongVideoBench","video_id":"lvb_Fo97qfO-9_g","filename":"Fo97qfO-9_g.mp4","archive":"videos.tar","member":"videos/Fo97qfO-9_g.mp4","bytes":68691297},
{"dataset":"LongVideoBench","video_id":"lvb_zda-T6wrEhs","filename":"zda-T6wrEhs.mp4","archive":"videos.tar","member":"videos/zda-T6wrEhs.mp4","bytes":18830226},
{"dataset":"LongVideoBench","video_id":"lvb_a_8G0PzVFbc","filename":"a_8G0PzVFbc.mp4","archive":"videos.tar","member":"videos/a_8G0PzVFbc.mp4","bytes":34543108},
{"dataset":"LongVideoBench","video_id":"lvb_HeRS3nwySI8","filename":"HeRS3nwySI8.mp4","archive":"videos.tar","member":"videos/HeRS3nwySI8.mp4","bytes":51640966},
{"dataset":"LongVideoBench","video_id":"lvb_d5JlCEDlHGE","filename":"d5JlCEDlHGE.mp4","archive":"videos.tar","member":"videos/d5JlCEDlHGE.mp4","bytes":46634624},
{"dataset":"LongVideoBench","video_id":"lvb_sHUEJw1BsGQ","filename":"sHUEJw1BsGQ.mp4","archive":"videos.tar","member":"videos/sHUEJw1BsGQ.mp4","bytes":94581379},
{"dataset":"LongVideoBench","video_id":"lvb_NMHmqgO04rU","filename":"NMHmqgO04rU.mp4","archive":"videos.tar","member":"videos/NMHmqgO04rU.mp4","bytes":276819656},
{"dataset":"LongVideoBench","video_id":"lvb_-eRimFrm6kQ","filename":"-eRimFrm6kQ.mp4","archive":"videos.tar","member":"videos/-eRimFrm6kQ.mp4","bytes":90220390},
{"dataset":"LongVideoBench","video_id":"lvb_@tiffycooks-7080578381569445125","filename":"7080578381569445125.mp4","archive":"videos.tar","member":"videos/7080578381569445125.mp4","bytes":4345786},
{"dataset":"LongVideoBench","video_id":"lvb_TynnVVRNZGs","filename":"TynnVVRNZGs.mp4","archive":"videos.tar","member":"videos/TynnVVRNZGs.mp4","bytes":5487075},
{"dataset":"LongVideoBench","video_id":"lvb_@thatrecipe.us-7266555380925287723","filename":"7266555380925287723.mp4","archive":"videos.tar","member":"videos/7266555380925287723.mp4","bytes":14771738},
{"dataset":"LongVideoBench","video_id":"lvb_HAED3riiZkw","filename":"HAED3riiZkw.mp4","archive":"videos.tar","member":"videos/HAED3riiZkw.mp4","bytes":25455471},
{"dataset":"LongVideoBench","video_id":"lvb_TlaX2iIYZD4","filename":"TlaX2iIYZD4.mp4","archive":"videos.tar","member":"videos/TlaX2iIYZD4.mp4","bytes":163692906},
{"dataset":"LongVideoBench","video_id":"lvb_XuQswmEPgxU","filename":"XuQswmEPgxU.mp4","archive":"videos.tar","member":"videos/XuQswmEPgxU.mp4","bytes":57757435},
{"dataset":"LongVideoBench","video_id":"lvb_@daiki.shino-7120297338056281349","filename":"7120297338056281349.mp4","archive":"videos.tar","member":"videos/7120297338056281349.mp4","bytes":1589602},
{"dataset":"LongVideoBench","video_id":"lvb_VSZ8ywgGNGM","filename":"VSZ8ywgGNGM.mp4","archive":"videos.tar","member":"videos/VSZ8ywgGNGM.mp4","bytes":9397365},
{"dataset":"LongVideoBench","video_id":"lvb_UN3ICsfqKEY","filename":"UN3ICsfqKEY.mp4","archive":"videos.tar","member":"videos/UN3ICsfqKEY.mp4","bytes":130429992},
{"dataset":"LongVideoBench","video_id":"lvb_to7vCdkLi4s","filename":"to7vCdkLi4s.mp4","archive":"videos.tar","member":"videos/to7vCdkLi4s.mp4","bytes":146627507},
{"dataset":"LongVideoBench","video_id":"lvb_M5YKW6fhlss","filename":"M5YKW6fhlss.mp4","archive":"videos.tar","member":"videos/M5YKW6fhlss.mp4","bytes":67451730},
{"dataset":"LongVideoBench","video_id":"lvb_4ouAf1ldH60","filename":"4ouAf1ldH60.mp4","archive":"videos.tar","member":"videos/4ouAf1ldH60.mp4","bytes":313044130},
{"dataset":"LongVideoBench","video_id":"lvb_eDso3zHFxL8","filename":"eDso3zHFxL8.mp4","archive":"videos.tar","member":"videos/eDso3zHFxL8.mp4","bytes":81659601},
{"dataset":"LongVideoBench","video_id":"lvb_mS1QPVgBDQo","filename":"mS1QPVgBDQo.mp4","archive":"videos.tar","member":"videos/mS1QPVgBDQo.mp4","bytes":334330784},
{"dataset":"LongVideoBench","video_id":"lvb_@lisolna-7321010343067618592","filename":"7321010343067618592.mp4","archive":"videos.tar","member":"videos/7321010343067618592.mp4","bytes":3691816},
{"dataset":"LongVideoBench","video_id":"lvb_XYsCVqz3iug","filename":"XYsCVqz3iug.mp4","archive":"videos.tar","member":"videos/XYsCVqz3iug.mp4","bytes":133465280},
{"dataset":"LongVideoBench","video_id":"lvb_8MkL3W6wU3g","filename":"8MkL3W6wU3g.mp4","archive":"videos.tar","member":"videos/8MkL3W6wU3g.mp4","bytes":255723449},
{"dataset":"LongVideoBench","video_id":"lvb_nvG2YMWiwqU","filename":"nvG2YMWiwqU.mp4","archive":"videos.tar","member":"videos/nvG2YMWiwqU.mp4","bytes":6805044},
{"dataset":"LongVideoBench","video_id":"lvb_OAcbasjxljY","filename":"OAcbasjxljY.mp4","archive":"videos.tar","member":"videos/OAcbasjxljY.mp4","bytes":91604669},
{"dataset":"LongVideoBench","video_id":"lvb_x4dFLY_-vKs","filename":"x4dFLY_-vKs.mp4","archive":"videos.tar","member":"videos/x4dFLY_-vKs.mp4","bytes":12261411},
{"dataset":"LongVideoBench","video_id":"lvb_wFrztzzohJ8","filename":"wFrztzzohJ8.mp4","archive":"videos.tar","member":"videos/wFrztzzohJ8.mp4","bytes":54367234},
{"dataset":"LongVideoBench","video_id":"lvb_VwZeSoYugZk","filename":"VwZeSoYugZk.mp4","archive":"videos.tar","member":"videos/VwZeSoYugZk.mp4","bytes":39884184},
{"dataset":"LongVideoBench","video_id":"lvb_rMXJOKhf_AA","filename":"rMXJOKhf_AA.mp4","archive":"videos.tar","member":"videos/rMXJOKhf_AA.mp4","bytes":18321386},
{"dataset":"LongVideoBench","video_id":"lvb_F2OhCCEIOcU","filename":"F2OhCCEIOcU.mp4","archive":"videos.tar","member":"videos/F2OhCCEIOcU.mp4","bytes":142865555},
{"dataset":"LongVideoBench","video_id":"lvb_7TljSpTBS9c","filename":"7TljSpTBS9c.mp4","archive":"videos.tar","member":"videos/7TljSpTBS9c.mp4","bytes":331395917},
{"dataset":"LongVideoBench","video_id":"lvb_571nruSayeo","filename":"571nruSayeo.mp4","archive":"videos.tar","member":"videos/571nruSayeo.mp4","bytes":175943103},
{"dataset":"LongVideoBench","video_id":"lvb_J1IwKg2ufk8","filename":"J1IwKg2ufk8.mp4","archive":"videos.tar","member":"videos/J1IwKg2ufk8.mp4","bytes":162578176},
{"dataset":"LongVideoBench","video_id":"lvb_O471uwTNx6k","filename":"O471uwTNx6k.mp4","archive":"videos.tar","member":"videos/O471uwTNx6k.mp4","bytes":87522732},
{"dataset":"LongVideoBench","video_id":"lvb_@healthfood-6999715804623293702","filename":"6999715804623293702.mp4","archive":"videos.tar","member":"videos/6999715804623293702.mp4","bytes":1665335},
{"dataset":"LongVideoBench","video_id":"lvb_dE5iWeCVpGI","filename":"dE5iWeCVpGI.mp4","archive":"videos.tar","member":"videos/dE5iWeCVpGI.mp4","bytes":72978021},
{"dataset":"LongVideoBench","video_id":"lvb_yl-6-Yzt--A","filename":"yl-6-Yzt--A.mp4","archive":"videos.tar","member":"videos/yl-6-Yzt--A.mp4","bytes":17547177},
{"dataset":"LongVideoBench","video_id":"lvb_WM78_KqcrSY","filename":"WM78_KqcrSY.mp4","archive":"videos.tar","member":"videos/WM78_KqcrSY.mp4","bytes":8537336},
{"dataset":"LongVideoBench","video_id":"lvb_AYMdAVxALP4","filename":"AYMdAVxALP4.mp4","archive":"videos.tar","member":"videos/AYMdAVxALP4.mp4","bytes":61646821},
{"dataset":"LongVideoBench","video_id":"lvb_X0U3fP0tZyY","filename":"X0U3fP0tZyY.mp4","archive":"videos.tar","member":"videos/X0U3fP0tZyY.mp4","bytes":45396822},
{"dataset":"LongVideoBench","video_id":"lvb_VFXJnbnN5ro","filename":"VFXJnbnN5ro.mp4","archive":"videos.tar","member":"videos/VFXJnbnN5ro.mp4","bytes":82104496},
{"dataset":"LongVideoBench","video_id":"lvb_GdFMKGNFXaE","filename":"GdFMKGNFXaE.mp4","archive":"videos.tar","member":"videos/GdFMKGNFXaE.mp4","bytes":145275201},
{"dataset":"LongVideoBench","video_id":"lvb_MJYBHfYF8LI","filename":"MJYBHfYF8LI.mp4","archive":"videos.tar","member":"videos/MJYBHfYF8LI.mp4","bytes":97151234},
{"dataset":"LongVideoBench","video_id":"lvb_zVudr8cxHRE","filename":"zVudr8cxHRE.mp4","archive":"videos.tar","member":"videos/zVudr8cxHRE.mp4","bytes":142257724},
{"dataset":"LongVideoBench","video_id":"lvb_7XWqI121-Q4","filename":"7XWqI121-Q4.mp4","archive":"videos.tar","member":"videos/7XWqI121-Q4.mp4","bytes":145891213},
{"dataset":"LongVideoBench","video_id":"lvb_uWBh0meTg08","filename":"uWBh0meTg08.mp4","archive":"videos.tar","member":"videos/uWBh0meTg08.mp4","bytes":12673101},
{"dataset":"LongVideoBench","video_id":"lvb_ydm72ftJStQ","filename":"ydm72ftJStQ.mp4","archive":"videos.tar","member":"videos/ydm72ftJStQ.mp4","bytes":6626310},
{"dataset":"LongVideoBench","video_id":"lvb_BktEeBeA7a8","filename":"BktEeBeA7a8.mp4","archive":"videos.tar","member":"videos/BktEeBeA7a8.mp4","bytes":77825525},
{"dataset":"LongVideoBench","video_id":"lvb_mH9LdC7IFH8","filename":"mH9LdC7IFH8.mp4","archive":"videos.tar","member":"videos/mH9LdC7IFH8.mp4","bytes":28638239},
{"dataset":"LongVideoBench","video_id":"lvb_Aau-XoIebno","filename":"Aau-XoIebno.mp4","archive":"videos.tar","member":"videos/Aau-XoIebno.mp4","bytes":227828335},
{"dataset":"LongVideoBench","video_id":"lvb_e_nAKiSutiA","filename":"e_nAKiSutiA.mp4","archive":"videos.tar","member":"videos/e_nAKiSutiA.mp4","bytes":791850},
{"dataset":"LongVideoBench","video_id":"lvb_Msz128EJeWE","filename":"Msz128EJeWE.mp4","archive":"videos.tar","member":"videos/Msz128EJeWE.mp4","bytes":1033994},
{"dataset":"LongVideoBench","video_id":"lvb_oddHY1vwcjo","filename":"oddHY1vwcjo.mp4","archive":"videos.tar","member":"videos/oddHY1vwcjo.mp4","bytes":1990597},
{"dataset":"LongVideoBench","video_id":"lvb_SO3czkzeFjw","filename":"SO3czkzeFjw.mp4","archive":"videos.tar","member":"videos/SO3czkzeFjw.mp4","bytes":41342034},
{"dataset":"LongVideoBench","video_id":"lvb_60oHeCZHtvI","filename":"60oHeCZHtvI.mp4","archive":"videos.tar","member":"videos/60oHeCZHtvI.mp4","bytes":41134622},
{"dataset":"LongVideoBench","video_id":"lvb_JASFwBtUK40","filename":"JASFwBtUK40.mp4","archive":"videos.tar","member":"videos/JASFwBtUK40.mp4","bytes":14242192},
{"dataset":"LongVideoBench","video_id":"lvb_QeXuzEwC9z4","filename":"QeXuzEwC9z4.mp4","archive":"videos.tar","member":"videos/QeXuzEwC9z4.mp4","bytes":7590062},
{"dataset":"LongVideoBench","video_id":"lvb_ZoUsR8t8IxE","filename":"ZoUsR8t8IxE.mp4","archive":"videos.tar","member":"videos/ZoUsR8t8IxE.mp4","bytes":49255266},
{"dataset":"LongVideoBench","video_id":"lvb_7F9IrtSHmc0","filename":"7F9IrtSHmc0.mp4","archive":"videos.tar","member":"videos/7F9IrtSHmc0.mp4","bytes":25862074},
{"dataset":"LongVideoBench","video_id":"lvb_GuEptwLiAvs","filename":"GuEptwLiAvs.mp4","archive":"videos.tar","member":"videos/GuEptwLiAvs.mp4","bytes":36286821},
{"dataset":"LongVideoBench","video_id":"lvb_q2bpkhjNxf0","filename":"q2bpkhjNxf0.mp4","archive":"videos.tar","member":"videos/q2bpkhjNxf0.mp4","bytes":515304},
{"dataset":"LongVideoBench","video_id":"lvb_QPth_xqBXGY","filename":"QPth_xqBXGY.mp4","archive":"videos.tar","member":"videos/QPth_xqBXGY.mp4","bytes":27781587},
{"dataset":"LongVideoBench","video_id":"lvb_GwcXYTX2GmA","filename":"GwcXYTX2GmA.mp4","archive":"videos.tar","member":"videos/GwcXYTX2GmA.mp4","bytes":1428483},
{"dataset":"LongVideoBench","video_id":"lvb_ip8khYCMb8Y","filename":"ip8khYCMb8Y.mp4","archive":"videos.tar","member":"videos/ip8khYCMb8Y.mp4","bytes":54409367},
{"dataset":"LongVideoBench","video_id":"lvb_ZGMGQsnSdLE","filename":"ZGMGQsnSdLE.mp4","archive":"videos.tar","member":"videos/ZGMGQsnSdLE.mp4","bytes":205739560},
{"dataset":"LongVideoBench","video_id":"lvb_z6THwql5c6w","filename":"z6THwql5c6w.mp4","archive":"videos.tar","member":"videos/z6THwql5c6w.mp4","bytes":63157226},
{"dataset":"LongVideoBench","video_id":"lvb_@placesunleashed-7289528053112343814","filename":"7289528053112343814.mp4","archive":"videos.tar","member":"videos/7289528053112343814.mp4","bytes":1837498},
{"dataset":"LongVideoBench","video_id":"lvb_KlZpZVphLrc","filename":"KlZpZVphLrc.mp4","archive":"videos.tar","member":"videos/KlZpZVphLrc.mp4","bytes":21760222},
{"dataset":"LongVideoBench","video_id":"lvb_mYotOV3Q51g","filename":"mYotOV3Q51g.mp4","archive":"videos.tar","member":"videos/mYotOV3Q51g.mp4","bytes":7177903},
{"dataset":"LongVideoBench","video_id":"lvb_LlqsCCa6y58","filename":"LlqsCCa6y58.mp4","archive":"videos.tar","member":"videos/LlqsCCa6y58.mp4","bytes":229884507},
{"dataset":"LongVideoBench","video_id":"lvb_s49y2RP5C7E","filename":"s49y2RP5C7E.mp4","archive":"videos.tar","member":"videos/s49y2RP5C7E.mp4","bytes":62932972},
{"dataset":"LongVideoBench","video_id":"lvb_athabNMGceo","filename":"athabNMGceo.mp4","archive":"videos.tar","member":"videos/athabNMGceo.mp4","bytes":29185206},
{"dataset":"LongVideoBench","video_id":"lvb_s2N4cPqCdys","filename":"s2N4cPqCdys.mp4","archive":"videos.tar","member":"videos/s2N4cPqCdys.mp4","bytes":5915448},
{"dataset":"LongVideoBench","video_id":"lvb_ZaXpMou55lw","filename":"ZaXpMou55lw.mp4","archive":"videos.tar","member":"videos/ZaXpMou55lw.mp4","bytes":45670673},
{"dataset":"LongVideoBench","video_id":"lvb_@movie.explained6-7275485247351901442","filename":"7275485247351901442.mp4","archive":"videos.tar","member":"videos/7275485247351901442.mp4","bytes":908113},
{"dataset":"LongVideoBench","video_id":"lvb_k194yU67Zik","filename":"k194yU67Zik.mp4","archive":"videos.tar","member":"videos/k194yU67Zik.mp4","bytes":3574075},
{"dataset":"LongVideoBench","video_id":"lvb_i327DBSS_iE","filename":"i327DBSS_iE.mp4","archive":"videos.tar","member":"videos/i327DBSS_iE.mp4","bytes":51197425},
{"dataset":"LongVideoBench","video_id":"lvb_9WjElCiDpzM","filename":"9WjElCiDpzM.mp4","archive":"videos.tar","member":"videos/9WjElCiDpzM.mp4","bytes":225551586},
{"dataset":"LongVideoBench","video_id":"lvb_58mCJVNMJvw","filename":"58mCJVNMJvw.mp4","archive":"videos.tar","member":"videos/58mCJVNMJvw.mp4","bytes":1038394},
{"dataset":"LongVideoBench","video_id":"lvb_@jetset_anna-7126584501228129542","filename":"7126584501228129542.mp4","archive":"videos.tar","member":"videos/7126584501228129542.mp4","bytes":1677923},
{"dataset":"LongVideoBench","video_id":"lvb_@recipesbyanne-7190395053343296774","filename":"7190395053343296774.mp4","archive":"videos.tar","member":"videos/7190395053343296774.mp4","bytes":1434494},
{"dataset":"LongVideoBench","video_id":"lvb_x4UBaEojM6U","filename":"x4UBaEojM6U.mp4","archive":"videos.tar","member":"videos/x4UBaEojM6U.mp4","bytes":115859951},
{"dataset":"LongVideoBench","video_id":"lvb_@tiffycooks-7171182579058511110","filename":"7171182579058511110.mp4","archive":"videos.tar","member":"videos/7171182579058511110.mp4","bytes":6196127},
{"dataset":"LongVideoBench","video_id":"lvb_tGiRbGGwRj8","filename":"tGiRbGGwRj8.mp4","archive":"videos.tar","member":"videos/tGiRbGGwRj8.mp4","bytes":13434008},
{"dataset":"LongVideoBench","video_id":"lvb_6hBbXVkgxGE","filename":"6hBbXVkgxGE.mp4","archive":"videos.tar","member":"videos/6hBbXVkgxGE.mp4","bytes":43085653},
{"dataset":"LongVideoBench","video_id":"lvb_YebUIUOCo94","filename":"YebUIUOCo94.mp4","archive":"videos.tar","member":"videos/YebUIUOCo94.mp4","bytes":127415702},
{"dataset":"LongVideoBench","video_id":"lvb_Y1YCvEip_ko","filename":"Y1YCvEip_ko.mp4","archive":"videos.tar","member":"videos/Y1YCvEip_ko.mp4","bytes":47223004},
{"dataset":"LongVideoBench","video_id":"lvb_tCnelzIAHA0","filename":"tCnelzIAHA0.mp4","archive":"videos.tar","member":"videos/tCnelzIAHA0.mp4","bytes":3825819},
{"dataset":"LongVideoBench","video_id":"lvb_fO7nwCix8xU","filename":"fO7nwCix8xU.mp4","archive":"videos.tar","member":"videos/fO7nwCix8xU.mp4","bytes":57826806},
{"dataset":"LongVideoBench","video_id":"lvb_lzuixOdhZow","filename":"lzuixOdhZow.mp4","archive":"videos.tar","member":"videos/lzuixOdhZow.mp4","bytes":2350639},
{"dataset":"LongVideoBench","video_id":"lvb_LAbtlJJhUlY","filename":"LAbtlJJhUlY.mp4","archive":"videos.tar","member":"videos/LAbtlJJhUlY.mp4","bytes":67665291},
{"dataset":"LongVideoBench","video_id":"lvb_o2F-N42Ufo4","filename":"o2F-N42Ufo4.mp4","archive":"videos.tar","member":"videos/o2F-N42Ufo4.mp4","bytes":113021476},
{"dataset":"LongVideoBench","video_id":"lvb_0_YDrJoUe8s","filename":"0_YDrJoUe8s.mp4","archive":"videos.tar","member":"videos/0_YDrJoUe8s.mp4","bytes":8138349},
{"dataset":"LongVideoBench","video_id":"lvb_@kelseyinlondon-7258968758130085146","filename":"7258968758130085146.mp4","archive":"videos.tar","member":"videos/7258968758130085146.mp4","bytes":2534197},
{"dataset":"LongVideoBench","video_id":"lvb_yz3lOAe32Tw","filename":"yz3lOAe32Tw.mp4","archive":"videos.tar","member":"videos/yz3lOAe32Tw.mp4","bytes":47372086},
{"dataset":"LongVideoBench","video_id":"lvb_b__dUom9AcQ","filename":"b__dUom9AcQ.mp4","archive":"videos.tar","member":"videos/b__dUom9AcQ.mp4","bytes":70923932},
{"dataset":"LongVideoBench","video_id":"lvb_@tiffycooks-7003765189401185542","filename":"7003765189401185542.mp4","archive":"videos.tar","member":"videos/7003765189401185542.mp4","bytes":8676132},
{"dataset":"LongVideoBench","video_id":"lvb_-PnG8Jp2gFw","filename":"-PnG8Jp2gFw.mp4","archive":"videos.tar","member":"videos/-PnG8Jp2gFw.mp4","bytes":46483208},
{"dataset":"LongVideoBench","video_id":"lvb_M-YfPangEfA","filename":"M-YfPangEfA.mp4","archive":"videos.tar","member":"videos/M-YfPangEfA.mp4","bytes":2861239},
{"dataset":"LongVideoBench","video_id":"lvb_iwXp1fT89-M","filename":"iwXp1fT89-M.mp4","archive":"videos.tar","member":"videos/iwXp1fT89-M.mp4","bytes":129103168},
{"dataset":"LongVideoBench","video_id":"lvb_ClYmTkGTGYg","filename":"ClYmTkGTGYg.mp4","archive":"videos.tar","member":"videos/ClYmTkGTGYg.mp4","bytes":254924573},
{"dataset":"LongVideoBench","video_id":"lvb_jiR7qHzgq_c","filename":"jiR7qHzgq_c.mp4","archive":"videos.tar","member":"videos/jiR7qHzgq_c.mp4","bytes":2099437},
{"dataset":"LongVideoBench","video_id":"lvb_@jess.morg-7255843192677911851","filename":"7255843192677911851.mp4","archive":"videos.tar","member":"videos/7255843192677911851.mp4","bytes":3425166},
{"dataset":"LongVideoBench","video_id":"lvb_wxWo44MoCTI","filename":"wxWo44MoCTI.mp4","archive":"videos.tar","member":"videos/wxWo44MoCTI.mp4","bytes":55822456},
{"dataset":"LongVideoBench","video_id":"lvb_Oht0i1DACcA","filename":"Oht0i1DACcA.mp4","archive":"videos.tar","member":"videos/Oht0i1DACcA.mp4","bytes":60935890},
{"dataset":"LongVideoBench","video_id":"lvb_XAAqKoQM8Qc","filename":"XAAqKoQM8Qc.mp4","archive":"videos.tar","member":"videos/XAAqKoQM8Qc.mp4","bytes":1278995},
{"dataset":"LongVideoBench","video_id":"lvb_AHeq99pojLo","filename":"AHeq99pojLo.mp4","archive":"videos.tar","member":"videos/AHeq99pojLo.mp4","bytes":2225381},
{"dataset":"LongVideoBench","video_id":"lvb_marOr6mJefE","filename":"marOr6mJefE.mp4","archive":"videos.tar","member":"videos/marOr6mJefE.mp4","bytes":54379858},
{"dataset":"LongVideoBench","video_id":"lvb_mB7NW91REF4","filename":"mB7NW91REF4.mp4","archive":"videos.tar","member":"videos/mB7NW91REF4.mp4","bytes":4542668},
{"dataset":"LongVideoBench","video_id":"lvb_-0aM99dMu_4","filename":"-0aM99dMu_4.mp4","archive":"videos.tar","member":"videos/-0aM99dMu_4.mp4","bytes":77279438},
{"dataset":"LongVideoBench","video_id":"lvb_tWiGnu2BNsY","filename":"tWiGnu2BNsY.mp4","archive":"videos.tar","member":"videos/tWiGnu2BNsY.mp4","bytes":29657155},
{"dataset":"LongVideoBench","video_id":"lvb_SKET024GdlE","filename":"SKET024GdlE.mp4","archive":"videos.tar","member":"videos/SKET024GdlE.mp4","bytes":1376479},
{"dataset":"LongVideoBench","video_id":"lvb_h4jIoMxZopU","filename":"h4jIoMxZopU.mp4","archive":"videos.tar","member":"videos/h4jIoMxZopU.mp4","bytes":18403395},
{"dataset":"Video-MME","video_id":"mme_819","filename":"GsYi-l5PEEs.mp4","archive":"videos_chunked_08.zip","member":"data/GsYi-l5PEEs.mp4","bytes":471521742},
{"dataset":"Video-MME","video_id":"mme_639","filename":"PHe0bXAIuk0.mp4","archive":"videos_chunked_13.zip","member":"data/PHe0bXAIuk0.mp4","bytes":83463381},
{"dataset":"Video-MME","video_id":"mme_609","filename":"HProiNnmGwI.mp4","archive":"videos_chunked_09.zip","member":"data/HProiNnmGwI.mp4","bytes":405248878},
{"dataset":"Video-MME","video_id":"mme_854","filename":"E83XxGvEPtI.mp4","archive":"videos_chunked_06.zip","member":"data/E83XxGvEPtI.mp4","bytes":177516832},
{"dataset":"Video-MME","video_id":"mme_694","filename":"Sp2nxlrQ89w.mp4","archive":"videos_chunked_15.zip","member":"data/Sp2nxlrQ89w.mp4","bytes":136659511},
{"dataset":"Video-MME","video_id":"mme_684","filename":"NCNeFSJexSY.mp4","archive":"videos_chunked_12.zip","member":"data/NCNeFSJexSY.mp4","bytes":103950872},
{"dataset":"Video-MME","video_id":"mme_677","filename":"MYxL_JLseC8.mp4","archive":"videos_chunked_12.zip","member":"data/MYxL_JLseC8.mp4","bytes":117239933},
{"dataset":"Video-MME","video_id":"mme_648","filename":"4iC9Qi3y9q8.mp4","archive":"videos_chunked_02.zip","member":"data/4iC9Qi3y9q8.mp4","bytes":96320346},
{"dataset":"Video-MME","video_id":"mme_852","filename":"eno3UMEMQJI.mp4","archive":"videos_chunked_07.zip","member":"data/eno3UMEMQJI.mp4","bytes":348523071},
{"dataset":"Video-MME","video_id":"mme_635","filename":"oue5A-7Hpx4.mp4","archive":"videos_chunked_13.zip","member":"data/oue5A-7Hpx4.mp4","bytes":188187882},
{"dataset":"Video-MME","video_id":"mme_831","filename":"6Pt3-NDxzRk.mp4","archive":"videos_chunked_03.zip","member":"data/6Pt3-NDxzRk.mp4","bytes":155283294},
{"dataset":"Video-MME","video_id":"mme_853","filename":"pKaPg9ZJ88Q.mp4","archive":"videos_chunked_13.zip","member":"data/pKaPg9ZJ88Q.mp4","bytes":269286097},
{"dataset":"Video-MME","video_id":"mme_787","filename":"FkrIcmXvv18.mp4","archive":"videos_chunked_07.zip","member":"data/FkrIcmXvv18.mp4","bytes":437980237},
{"dataset":"Video-MME","video_id":"mme_630","filename":"BjmPvovl-V4.mp4","archive":"videos_chunked_05.zip","member":"data/BjmPvovl-V4.mp4","bytes":173613738},
{"dataset":"Video-MME","video_id":"mme_802","filename":"TYmHe20p_zU.mp4","archive":"videos_chunked_16.zip","member":"data/TYmHe20p_zU.mp4","bytes":192063366},
{"dataset":"Video-MME","video_id":"mme_745","filename":"mRgstQ4X5d4.mp4","archive":"videos_chunked_12.zip","member":"data/mRgstQ4X5d4.mp4","bytes":613827987},
{"dataset":"Video-MME","video_id":"mme_611","filename":"H8fGd3fCJbg.mp4","archive":"videos_chunked_09.zip","member":"data/H8fGd3fCJbg.mp4","bytes":90711310},
{"dataset":"Video-MME","video_id":"mme_632","filename":"FQd5bo9nIZs.mp4","archive":"videos_chunked_07.zip","member":"data/FQd5bo9nIZs.mp4","bytes":134474191},
{"dataset":"Video-MME","video_id":"mme_675","filename":"lVeNTofDB2k.mp4","archive":"videos_chunked_11.zip","member":"data/lVeNTofDB2k.mp4","bytes":104695961},
{"dataset":"Video-MME","video_id":"mme_680","filename":"y2kg3MOk1sY.mp4","archive":"videos_chunked_18.zip","member":"data/y2kg3MOk1sY.mp4","bytes":316734991},
{"dataset":"Video-MME","video_id":"mme_773","filename":"BrjwEoPXyxA.mp4","archive":"videos_chunked_05.zip","member":"data/BrjwEoPXyxA.mp4","bytes":736021647},
{"dataset":"Video-MME","video_id":"mme_806","filename":"r1PoSdFWvQg.mp4","archive":"videos_chunked_14.zip","member":"data/r1PoSdFWvQg.mp4","bytes":183178780},
{"dataset":"Video-MME","video_id":"mme_610","filename":"O7zcPvQDIMU.mp4","archive":"videos_chunked_12.zip","member":"data/O7zcPvQDIMU.mp4","bytes":229171699},
{"dataset":"Video-MME","video_id":"mme_792","filename":"0bwyCF0HteE.mp4","archive":"videos_chunked_01.zip","member":"data/0bwyCF0HteE.mp4","bytes":219217274},
{"dataset":"Video-MME","video_id":"mme_668","filename":"C6aeL83z_9Y.mp4","archive":"videos_chunked_05.zip","member":"data/C6aeL83z_9Y.mp4","bytes":166269470},
{"dataset":"Video-MME","video_id":"mme_845","filename":"1evyOuQz-jM.mp4","archive":"videos_chunked_01.zip","member":"data/1evyOuQz-jM.mp4","bytes":150758326},
{"dataset":"Video-MME","video_id":"mme_822","filename":"3MJNKd10kxM.mp4","archive":"videos_chunked_02.zip","member":"data/3MJNKd10kxM.mp4","bytes":412327443},
{"dataset":"Video-MME","video_id":"mme_840","filename":"y9Fne3oUwX4.mp4","archive":"videos_chunked_18.zip","member":"data/y9Fne3oUwX4.mp4","bytes":595937589},
{"dataset":"Video-MME","video_id":"mme_896","filename":"xIWaK92gRlo.mp4","archive":"videos_chunked_18.zip","member":"data/xIWaK92gRlo.mp4","bytes":596540790},
{"dataset":"Video-MME","video_id":"mme_744","filename":"4od5mdtEp-E.mp4","archive":"videos_chunked_02.zip","member":"data/4od5mdtEp-E.mp4","bytes":567757401},
{"dataset":"Video-MME","video_id":"mme_676","filename":"YPzzGPXW8fs.mp4","archive":"videos_chunked_19.zip","member":"data/YPzzGPXW8fs.mp4","bytes":401869925},
{"dataset":"Video-MME","video_id":"mme_754","filename":"0k2ey_okQ4E.mp4","archive":"videos_chunked_01.zip","member":"data/0k2ey_okQ4E.mp4","bytes":402924912},
{"dataset":"Video-MME","video_id":"mme_695","filename":"AnWKOvjFl8k.mp4","archive":"videos_chunked_04.zip","member":"data/AnWKOvjFl8k.mp4","bytes":160195276},
{"dataset":"Video-MME","video_id":"mme_877","filename":"5WIdIs3A9Ok.mp4","archive":"videos_chunked_03.zip","member":"data/5WIdIs3A9Ok.mp4","bytes":345124687},
{"dataset":"Video-MME","video_id":"mme_603","filename":"7D-gxaie6UI.mp4","archive":"videos_chunked_03.zip","member":"data/7D-gxaie6UI.mp4","bytes":158818600},
{"dataset":"Video-MME","video_id":"mme_860","filename":"K37YKK9h9pA.mp4","archive":"videos_chunked_10.zip","member":"data/K37YKK9h9pA.mp4","bytes":203468878},
{"dataset":"Video-MME","video_id":"mme_876","filename":"whksDmTR9YE.mp4","archive":"videos_chunked_17.zip","member":"data/whksDmTR9YE.mp4","bytes":361901967},
{"dataset":"Video-MME","video_id":"mme_655","filename":"PXxscnWG8QA.mp4","archive":"videos_chunked_13.zip","member":"data/PXxscnWG8QA.mp4","bytes":293636770},
{"dataset":"Video-MME","video_id":"mme_839","filename":"azJ5pk5reX0.mp4","archive":"videos_chunked_04.zip","member":"data/azJ5pk5reX0.mp4","bytes":193654267},
{"dataset":"Video-MME","video_id":"mme_895","filename":"1O1TfTrEnss.mp4","archive":"videos_chunked_01.zip","member":"data/1O1TfTrEnss.mp4","bytes":555959857},
{"dataset":"Video-MME","video_id":"mme_717","filename":"068rdc75mHM.mp4","archive":"videos_chunked_01.zip","member":"data/068rdc75mHM.mp4","bytes":194379356},
{"dataset":"Video-MME","video_id":"mme_889","filename":"EP2rD6lr2Xk.mp4","archive":"videos_chunked_07.zip","member":"data/EP2rD6lr2Xk.mp4","bytes":339625671},
{"dataset":"Video-MME","video_id":"mme_654","filename":"HvKpnaXYUPU.mp4","archive":"videos_chunked_09.zip","member":"data/HvKpnaXYUPU.mp4","bytes":253593229},
{"dataset":"Video-MME","video_id":"mme_674","filename":"gKjpYd1Ogwg.mp4","archive":"videos_chunked_08.zip","member":"data/gKjpYd1Ogwg.mp4","bytes":156889371},
{"dataset":"Video-MME","video_id":"mme_861","filename":"_T2Avd3tFHc.mp4","archive":"videos_chunked_15.zip","member":"data/_T2Avd3tFHc.mp4","bytes":271615396},
{"dataset":"Video-MME","video_id":"mme_715","filename":"9edWX7TTsLw.mp4","archive":"videos_chunked_04.zip","member":"data/9edWX7TTsLw.mp4","bytes":88118332},
{"dataset":"Video-MME","video_id":"mme_897","filename":"nO2B4haj2BQ.mp4","archive":"videos_chunked_12.zip","member":"data/nO2B4haj2BQ.mp4","bytes":397148854},
{"dataset":"Video-MME","video_id":"mme_730","filename":"1NH5dJ9VRvU.mp4","archive":"videos_chunked_01.zip","member":"data/1NH5dJ9VRvU.mp4","bytes":462023363},
{"dataset":"Video-MME","video_id":"mme_634","filename":"0ncLRRLvdfk.mp4","archive":"videos_chunked_01.zip","member":"data/0ncLRRLvdfk.mp4","bytes":114017212},
{"dataset":"Video-MME","video_id":"mme_723","filename":"n5DXniZGkME.mp4","archive":"videos_chunked_12.zip","member":"data/n5DXniZGkME.mp4","bytes":143693843},
{"dataset":"Video-MME","video_id":"mme_718","filename":"eAIKvD_gLJo.mp4","archive":"videos_chunked_06.zip","member":"data/eAIKvD_gLJo.mp4","bytes":202363284},
{"dataset":"Video-MME","video_id":"mme_807","filename":"dLjOLXmu68M.mp4","archive":"videos_chunked_06.zip","member":"data/dLjOLXmu68M.mp4","bytes":155582017},
{"dataset":"Video-MME","video_id":"mme_691","filename":"lMxFbRc3Luk.mp4","archive":"videos_chunked_11.zip","member":"data/lMxFbRc3Luk.mp4","bytes":137939384},
{"dataset":"Video-MME","video_id":"mme_615","filename":"LCfBYE97rFk.mp4","archive":"videos_chunked_11.zip","member":"data/LCfBYE97rFk.mp4","bytes":187502814},
{"dataset":"Video-MME","video_id":"mme_850","filename":"5Z2_uRf7ckY.mp4","archive":"videos_chunked_03.zip","member":"data/5Z2_uRf7ckY.mp4","bytes":208548944},
{"dataset":"Video-MME","video_id":"mme_757","filename":"jdQ-20JEmgc.mp4","archive":"videos_chunked_10.zip","member":"data/jdQ-20JEmgc.mp4","bytes":342856013},
{"dataset":"Video-MME","video_id":"mme_784","filename":"FvZFBZB10LA.mp4","archive":"videos_chunked_08.zip","member":"data/FvZFBZB10LA.mp4","bytes":359533893},
{"dataset":"Video-MME","video_id":"mme_643","filename":"Ei-TcECJVXU.mp4","archive":"videos_chunked_06.zip","member":"data/Ei-TcECJVXU.mp4","bytes":191800956},
{"dataset":"Video-MME","video_id":"mme_627","filename":"yQ6VOOd73MA.mp4","archive":"videos_chunked_19.zip","member":"data/yQ6VOOd73MA.mp4","bytes":201700825},
{"dataset":"Video-MME","video_id":"mme_789","filename":"i7ivyMFsw-w.mp4","archive":"videos_chunked_09.zip","member":"data/i7ivyMFsw-w.mp4","bytes":138957628},
{"dataset":"Video-MME","video_id":"mme_701","filename":"-dfvdKf-KR0.mp4","archive":"videos_chunked_06.zip","member":"data/-dfvdKf-KR0.mp4","bytes":168377723},
{"dataset":"Video-MME","video_id":"mme_815","filename":"7TydWUguPRU.mp4","archive":"videos_chunked_03.zip","member":"data/7TydWUguPRU.mp4","bytes":234258314},
{"dataset":"Video-MME","video_id":"mme_812","filename":"yh-EHgkFci4.mp4","archive":"videos_chunked_19.zip","member":"data/yh-EHgkFci4.mp4","bytes":230486612},
{"dataset":"Video-MME","video_id":"mme_724","filename":"54k2g9ddy4M.mp4","archive":"videos_chunked_02.zip","member":"data/54k2g9ddy4M.mp4","bytes":424403085},
{"dataset":"Video-MME","video_id":"mme_798","filename":"k2FIFQIYBvA.mp4","archive":"videos_chunked_10.zip","member":"data/k2FIFQIYBvA.mp4","bytes":141472445},
{"dataset":"Video-MME","video_id":"mme_666","filename":"mzvHYBgD-tI.mp4","archive":"videos_chunked_12.zip","member":"data/mzvHYBgD-tI.mp4","bytes":82173798},
{"dataset":"Video-MME","video_id":"mme_841","filename":"WB4giHwiulE.mp4","archive":"videos_chunked_17.zip","member":"data/WB4giHwiulE.mp4","bytes":348599627},
{"dataset":"Video-MME","video_id":"mme_624","filename":"mAwgdX5VxGc.mp4","archive":"videos_chunked_11.zip","member":"data/mAwgdX5VxGc.mp4","bytes":77087865},
{"dataset":"Video-MME","video_id":"mme_616","filename":"aansXcMqnNk.mp4","archive":"videos_chunked_04.zip","member":"data/aansXcMqnNk.mp4","bytes":245922408},
{"dataset":"Video-MME","video_id":"mme_826","filename":"dXBREPP1hfo.mp4","archive":"videos_chunked_06.zip","member":"data/dXBREPP1hfo.mp4","bytes":90592670},
{"dataset":"Video-MME","video_id":"mme_678","filename":"B0-7MeWA2aI.mp4","archive":"videos_chunked_04.zip","member":"data/B0-7MeWA2aI.mp4","bytes":80701687},
{"dataset":"Video-MME","video_id":"mme_864","filename":"VFntoBRGF1A.mp4","archive":"videos_chunked_17.zip","member":"data/VFntoBRGF1A.mp4","bytes":630969202},
{"dataset":"Video-MME","video_id":"mme_699","filename":"I2dv1jNM7gM.mp4","archive":"videos_chunked_09.zip","member":"data/I2dv1jNM7gM.mp4","bytes":133048359},
{"dataset":"Video-MME","video_id":"mme_628","filename":"YdMCL9_UTE4.mp4","archive":"videos_chunked_19.zip","member":"data/YdMCL9_UTE4.mp4","bytes":156069772},
{"dataset":"Video-MME","video_id":"mme_894","filename":"tslKtm6Le1s.mp4","archive":"videos_chunked_16.zip","member":"data/tslKtm6Le1s.mp4","bytes":196958173},
{"dataset":"Video-MME","video_id":"mme_884","filename":"0kRsiSdDFYg.mp4","archive":"videos_chunked_01.zip","member":"data/0kRsiSdDFYg.mp4","bytes":465957470},
{"dataset":"Video-MME","video_id":"mme_886","filename":"elprD1hnDyU.mp4","archive":"videos_chunked_06.zip","member":"data/elprD1hnDyU.mp4","bytes":200482755},
{"dataset":"Video-MME","video_id":"mme_755","filename":"HGv87QeBYnc.mp4","archive":"videos_chunked_09.zip","member":"data/HGv87QeBYnc.mp4","bytes":253747601},
{"dataset":"Video-MME","video_id":"mme_817","filename":"EuBhksmj0sQ.mp4","archive":"videos_chunked_07.zip","member":"data/EuBhksmj0sQ.mp4","bytes":528668261},
{"dataset":"Video-MME","video_id":"mme_725","filename":"p_4UPdFqgIQ.mp4","archive":"videos_chunked_13.zip","member":"data/p_4UPdFqgIQ.mp4","bytes":135006002},
{"dataset":"Video-MME","video_id":"mme_656","filename":"iOjmy8zkDGQ.mp4","archive":"videos_chunked_10.zip","member":"data/iOjmy8zkDGQ.mp4","bytes":524786185},
{"dataset":"Video-MME","video_id":"mme_727","filename":"yk4eWjYVNYg.mp4","archive":"videos_chunked_19.zip","member":"data/yk4eWjYVNYg.mp4","bytes":178894537},
{"dataset":"Video-MME","video_id":"mme_722","filename":"54H8ppxnp8I.mp4","archive":"videos_chunked_02.zip","member":"data/54H8ppxnp8I.mp4","bytes":109960794},
{"dataset":"Video-MME","video_id":"mme_672","filename":"Vu_NnDWxKY4.mp4","archive":"videos_chunked_17.zip","member":"data/Vu_NnDWxKY4.mp4","bytes":65968992},
{"dataset":"Video-MME","video_id":"mme_829","filename":"h5Jx12SYKzk.mp4","archive":"videos_chunked_08.zip","member":"data/h5Jx12SYKzk.mp4","bytes":201295482},
{"dataset":"Video-MME","video_id":"mme_692","filename":"TF9I1GxNdJQ.mp4","archive":"videos_chunked_15.zip","member":"data/TF9I1GxNdJQ.mp4","bytes":117773708},
{"dataset":"Video-MME","video_id":"mme_891","filename":"uuCVnqV4cNc.mp4","archive":"videos_chunked_16.zip","member":"data/uuCVnqV4cNc.mp4","bytes":233097342},
{"dataset":"Video-MME","video_id":"mme_834","filename":"aqFfjJrLkBA.mp4","archive":"videos_chunked_04.zip","member":"data/aqFfjJrLkBA.mp4","bytes":328751383},
{"dataset":"Video-MME","video_id":"mme_625","filename":"iCQmfRMwHfA.mp4","archive":"videos_chunked_09.zip","member":"data/iCQmfRMwHfA.mp4","bytes":57961604},
{"dataset":"Video-MME","video_id":"mme_808","filename":"leG0U5wzGlA.mp4","archive":"videos_chunked_11.zip","member":"data/leG0U5wzGlA.mp4","bytes":155685039},
{"dataset":"Video-MME","video_id":"mme_872","filename":"ZP-Y2S750KY.mp4","archive":"videos_chunked_20.zip","member":"data/ZP-Y2S750KY.mp4","bytes":148926701},
{"dataset":"Video-MME","video_id":"mme_659","filename":"K946mptgzTU.mp4","archive":"videos_chunked_10.zip","member":"data/K946mptgzTU.mp4","bytes":295283660},
{"dataset":"Video-MME","video_id":"mme_783","filename":"liDQtpusSmY.mp4","archive":"videos_chunked_11.zip","member":"data/liDQtpusSmY.mp4","bytes":298198004},
{"dataset":"Video-MME","video_id":"mme_849","filename":"j27UP4zz_6U.mp4","archive":"videos_chunked_10.zip","member":"data/j27UP4zz_6U.mp4","bytes":153896550},
{"dataset":"Video-MME","video_id":"mme_899","filename":"f-1xE_5uXDg.mp4","archive":"videos_chunked_07.zip","member":"data/f-1xE_5uXDg.mp4","bytes":145392764},
{"dataset":"Video-MME","video_id":"mme_758","filename":"rhDdA-7gEhs.mp4","archive":"videos_chunked_14.zip","member":"data/rhDdA-7gEhs.mp4","bytes":261451205},
{"dataset":"Video-MME","video_id":"mme_693","filename":"dXUXfyiX2lA.mp4","archive":"videos_chunked_06.zip","member":"data/dXUXfyiX2lA.mp4","bytes":145368678},
{"dataset":"Video-MME","video_id":"mme_835","filename":"IXjkbgrdAkw.mp4","archive":"videos_chunked_10.zip","member":"data/IXjkbgrdAkw.mp4","bytes":302971126},
{"dataset":"Video-MME","video_id":"mme_791","filename":"9tBsMSDoDqk.mp4","archive":"videos_chunked_04.zip","member":"data/9tBsMSDoDqk.mp4","bytes":187875370},
{"dataset":"Video-MME","video_id":"mme_711","filename":"XOfeqRgaF1A.mp4","archive":"videos_chunked_18.zip","member":"data/XOfeqRgaF1A.mp4","bytes":156547722},
{"dataset":"Video-MME","video_id":"mme_863","filename":"rSnQR7vGqMY.mp4","archive":"videos_chunked_14.zip","member":"data/rSnQR7vGqMY.mp4","bytes":338485038},
{"dataset":"Video-MME","video_id":"mme_620","filename":"mOiEOs3ZlT8.mp4","archive":"videos_chunked_11.zip","member":"data/mOiEOs3ZlT8.mp4","bytes":93653314},
{"dataset":"Video-MME","video_id":"mme_679","filename":"z7SoAIq20lA.mp4","archive":"videos_chunked_19.zip","member":"data/z7SoAIq20lA.mp4","bytes":197701615},
{"dataset":"Video-MME","video_id":"mme_708","filename":"xGcfBRkJSWQ.mp4","archive":"videos_chunked_18.zip","member":"data/xGcfBRkJSWQ.mp4","bytes":170637278},
{"dataset":"Video-MME","video_id":"mme_737","filename":"HT2iYl6E_Vo.mp4","archive":"videos_chunked_09.zip","member":"data/HT2iYl6E_Vo.mp4","bytes":783157714},
{"dataset":"Video-MME","video_id":"mme_623","filename":"gfikT_O4v9A.mp4","archive":"videos_chunked_08.zip","member":"data/gfikT_O4v9A.mp4","bytes":78522913},
{"dataset":"Video-MME","video_id":"mme_673","filename":"0Jbc3Ah4EIc.mp4","archive":"videos_chunked_01.zip","member":"data/0Jbc3Ah4EIc.mp4","bytes":116256178},
{"dataset":"Video-MME","video_id":"mme_794","filename":"XDq08lk5GnQ.mp4","archive":"videos_chunked_18.zip","member":"data/XDq08lk5GnQ.mp4","bytes":195664207},
{"dataset":"Video-MME","video_id":"mme_846","filename":"fLO5Ww0V2qU.mp4","archive":"videos_chunked_07.zip","member":"data/fLO5Ww0V2qU.mp4","bytes":537222788},
{"dataset":"Video-MME","video_id":"mme_824","filename":"p0qf48lXD4M.mp4","archive":"videos_chunked_13.zip","member":"data/p0qf48lXD4M.mp4","bytes":335651023},
{"dataset":"Video-MME","video_id":"mme_771","filename":"Bgh-2z6P0ao.mp4","archive":"videos_chunked_05.zip","member":"data/Bgh-2z6P0ao.mp4","bytes":434074859},
{"dataset":"Video-MME","video_id":"mme_736","filename":"-y1U0qk1vDs.mp4","archive":"videos_chunked_18.zip","member":"data/-y1U0qk1vDs.mp4","bytes":428155031},
{"dataset":"Video-MME","video_id":"mme_820","filename":"SeDGkRfpTqQ.mp4","archive":"videos_chunked_15.zip","member":"data/SeDGkRfpTqQ.mp4","bytes":372674082},
{"dataset":"Video-MME","video_id":"mme_649","filename":"mvpbZGwZ6_4.mp4","archive":"videos_chunked_12.zip","member":"data/mvpbZGwZ6_4.mp4","bytes":190459751},
{"dataset":"Video-MME","video_id":"mme_898","filename":"K9MQATj3894.mp4","archive":"videos_chunked_10.zip","member":"data/K9MQATj3894.mp4","bytes":546490066},
{"dataset":"Video-MME","video_id":"mme_685","filename":"nVtb2vNUOdU.mp4","archive":"videos_chunked_12.zip","member":"data/nVtb2vNUOdU.mp4","bytes":131123530},
{"dataset":"Video-MME","video_id":"mme_855","filename":"vzfTpidE5wg.mp4","archive":"videos_chunked_17.zip","member":"data/vzfTpidE5wg.mp4","bytes":141135510},
{"dataset":"Video-MME","video_id":"mme_892","filename":"8UQFaVCV8S8.mp4","archive":"videos_chunked_04.zip","member":"data/8UQFaVCV8S8.mp4","bytes":658800006},
{"dataset":"Video-MME","video_id":"mme_690","filename":"b0HfmY64eSE.mp4","archive":"videos_chunked_04.zip","member":"data/b0HfmY64eSE.mp4","bytes":195578222},
{"dataset":"Video-MME","video_id":"mme_800","filename":"Cb91B0wZYLE.mp4","archive":"videos_chunked_05.zip","member":"data/Cb91B0wZYLE.mp4","bytes":178794631},
{"dataset":"Video-MME","video_id":"mme_747","filename":"4qYqPmIO0v0.mp4","archive":"videos_chunked_02.zip","member":"data/4qYqPmIO0v0.mp4","bytes":914492603},
{"dataset":"MLVU","video_id":"mlvu_en_tv_28","filename":"en_tv_28.mp4","archive":"video_part_6.zip","member":"en_tv_28.mp4","bytes":112888559},
{"dataset":"MLVU","video_id":"mlvu_needle_40","filename":"needle_40.mp4","archive":"video_part_4.zip","member":"needle_40.mp4","bytes":44418409},
{"dataset":"MLVU","video_id":"mlvu_order_128","filename":"order_128.mp4","archive":"video_part_8.zip","member":"order_128.mp4","bytes":93655950},
{"dataset":"MLVU","video_id":"mlvu_game_42","filename":"game_42.mp4","archive":"video_part_3.zip","member":"game_42.mp4","bytes":460002726},
{"dataset":"MLVU","video_id":"mlvu_232","filename":"232.mp4","archive":"video_part_6.zip","member":"232.mp4","bytes":130667057},
{"dataset":"MLVU","video_id":"mlvu_xiaoliyu_5","filename":"xiaoliyu_5.mp4","archive":"video_part_2.zip","member":"xiaoliyu_5.mp4","bytes":391506001},
{"dataset":"MLVU","video_id":"mlvu_movie101_57","filename":"movie101_57.mp4","archive":"video_part_8.zip","member":"movie101_57.mp4","bytes":222129725},
{"dataset":"MLVU","video_id":"mlvu_haimian_10","filename":"haimian_10.mp4","archive":"video_part_1.zip","member":"haimian_10.mp4","bytes":44982320},
{"dataset":"MLVU","video_id":"mlvu_count_67","filename":"count_67.mp4","archive":"video_part_2.zip","member":"count_67.mp4","bytes":71310821},
{"dataset":"MLVU","video_id":"mlvu_xiaoliyu_6","filename":"xiaoliyu_6.mp4","archive":"video_part_2.zip","member":"xiaoliyu_6.mp4","bytes":337137889},
{"dataset":"MLVU","video_id":"mlvu_ego_72","filename":"ego_72.mp4","archive":"video_part_4.zip","member":"ego_72.mp4","bytes":199226237},
{"dataset":"MLVU","video_id":"mlvu_en_tv_33","filename":"en_tv_33.mp4","archive":"video_part_1.zip","member":"en_tv_33.mp4","bytes":258747617},
{"dataset":"MLVU","video_id":"mlvu_en_tv_12","filename":"en_tv_12.mp4","archive":"video_part_8.zip","member":"en_tv_12.mp4","bytes":306837215},
{"dataset":"MLVU","video_id":"mlvu_ego_14","filename":"ego_14.mp4","archive":"video_part_2.zip","member":"ego_14.mp4","bytes":358684633},
{"dataset":"MLVU","video_id":"mlvu_ego_16","filename":"ego_16.mp4","archive":"video_part_4.zip","member":"ego_16.mp4","bytes":93355444},
{"dataset":"MLVU","video_id":"mlvu_ego_5","filename":"ego_5.mp4","archive":"video_part_5.zip","member":"ego_5.mp4","bytes":285830888},
{"dataset":"MLVU","video_id":"mlvu_206","filename":"206.mp4","archive":"video_part_3.zip","member":"206.mp4","bytes":129257043},
{"dataset":"MLVU","video_id":"mlvu_count_102","filename":"count_102.mp4","archive":"video_part_7.zip","member":"count_102.mp4","bytes":152416607},
{"dataset":"MLVU","video_id":"mlvu_needle_22","filename":"needle_22.mp4","archive":"video_part_3.zip","member":"needle_22.mp4","bytes":89357672},
{"dataset":"MLVU","video_id":"mlvu_count_143","filename":"count_143.mp4","archive":"video_part_1.zip","member":"count_143.mp4","bytes":67099056},
{"dataset":"MLVU","video_id":"mlvu_order_97","filename":"order_97.mp4","archive":"video_part_1.zip","member":"order_97.mp4","bytes":94054814},
{"dataset":"MLVU","video_id":"mlvu_ego_71","filename":"ego_71.mp4","archive":"video_part_2.zip","member":"ego_71.mp4","bytes":524253928},
{"dataset":"MLVU","video_id":"mlvu_en_tv_21","filename":"en_tv_21.mp4","archive":"video_part_7.zip","member":"en_tv_21.mp4","bytes":101097720},
{"dataset":"MLVU","video_id":"mlvu_needle_98","filename":"needle_98.mp4","archive":"video_part_4.zip","member":"needle_98.mp4","bytes":54792563},
{"dataset":"MLVU","video_id":"mlvu_needle_48","filename":"needle_48.mp4","archive":"video_part_2.zip","member":"needle_48.mp4","bytes":76782518},
{"dataset":"MLVU","video_id":"mlvu_order_311","filename":"order_311.mp4","archive":"video_part_2.zip","member":"order_311.mp4","bytes":64657063},
{"dataset":"MLVU","video_id":"mlvu_en_tv_30","filename":"en_tv_30.mp4","archive":"video_part_3.zip","member":"en_tv_30.mp4","bytes":266492620},
{"dataset":"MLVU","video_id":"mlvu_tomjerry_9","filename":"tomjerry_9.mp4","archive":"video_part_6.zip","member":"tomjerry_9.mp4","bytes":35813212},
{"dataset":"MLVU","video_id":"mlvu_needle_92","filename":"needle_92.mp4","archive":"video_part_1.zip","member":"needle_92.mp4","bytes":66088991},
{"dataset":"MLVU","video_id":"mlvu_tomjerry_11","filename":"tomjerry_11.mp4","archive":"video_part_5.zip","member":"tomjerry_11.mp4","bytes":49911199},
{"dataset":"MLVU","video_id":"mlvu_movie101_55","filename":"movie101_55.mp4","archive":"video_part_3.zip","member":"movie101_55.mp4","bytes":108854053},
{"dataset":"MLVU","video_id":"mlvu_surveil_68","filename":"surveil_68.mp4","archive":"video_part_2.zip","member":"surveil_68.mp4","bytes":60444556},
{"dataset":"MLVU","video_id":"mlvu_haimian_1","filename":"haimian_1.mp4","archive":"video_part_2.zip","member":"haimian_1.mp4","bytes":44813998},
{"dataset":"MLVU","video_id":"mlvu_surveil_165","filename":"surveil_165.mp4","archive":"video_part_2.zip","member":"surveil_165.mp4","bytes":48950621},
{"dataset":"MLVU","video_id":"mlvu_needle_68","filename":"needle_68.mp4","archive":"video_part_2.zip","member":"needle_68.mp4","bytes":144961787},
{"dataset":"MLVU","video_id":"mlvu_order_264","filename":"order_264.mp4","archive":"video_part_4.zip","member":"order_264.mp4","bytes":47327983},
{"dataset":"MLVU","video_id":"mlvu_1","filename":"1.mp4","archive":"video_part_7.zip","member":"1.mp4","bytes":25940350},
{"dataset":"MLVU","video_id":"mlvu_count_149","filename":"count_149.mp4","archive":"video_part_6.zip","member":"count_149.mp4","bytes":71584579},
{"dataset":"MLVU","video_id":"mlvu_movie101_8","filename":"movie101_8.mp4","archive":"video_part_5.zip","member":"movie101_8.mp4","bytes":210315418},
{"dataset":"MLVU","video_id":"mlvu_surveil_152","filename":"surveil_152.mp4","archive":"video_part_4.zip","member":"surveil_152.mp4","bytes":73407462},
{"dataset":"MLVU","video_id":"mlvu_movie101_30","filename":"movie101_30.mp4","archive":"video_part_3.zip","member":"movie101_30.mp4","bytes":190797158},
{"dataset":"MLVU","video_id":"mlvu_order_342","filename":"order_342.mp4","archive":"video_part_4.zip","member":"order_342.mp4","bytes":54015403},
{"dataset":"MLVU","video_id":"mlvu_surveil_45","filename":"surveil_45.mp4","archive":"video_part_2.zip","member":"surveil_45.mp4","bytes":79840060},
{"dataset":"MLVU","video_id":"mlvu_AWE-4","filename":"AWE-4.mp4","archive":"video_part_6.zip","member":"AWE-4.mp4","bytes":70869727},
{"dataset":"MLVU","video_id":"mlvu_movie101_83","filename":"movie101_83.mp4","archive":"video_part_8.zip","member":"movie101_83.mp4","bytes":181308041},
{"dataset":"MLVU","video_id":"mlvu_movie101_108","filename":"movie101_108.mp4","archive":"video_part_5.zip","member":"movie101_108.mp4","bytes":97718429},
{"dataset":"MLVU","video_id":"mlvu_order_86","filename":"order_86.mp4","archive":"video_part_2.zip","member":"order_86.mp4","bytes":72405524},
{"dataset":"MLVU","video_id":"mlvu_ego_3","filename":"ego_3.mp4","archive":"video_part_2.zip","member":"ego_3.mp4","bytes":240541465},
{"dataset":"MLVU","video_id":"mlvu_needle_36","filename":"needle_36.mp4","archive":"video_part_7.zip","member":"needle_36.mp4","bytes":81182593},
{"dataset":"MLVU","video_id":"mlvu_count_34","filename":"count_34.mp4","archive":"video_part_3.zip","member":"count_34.mp4","bytes":56632624},
{"dataset":"MLVU","video_id":"mlvu_surveil_173","filename":"surveil_173.mp4","archive":"video_part_5.zip","member":"surveil_173.mp4","bytes":57221709},
{"dataset":"MLVU","video_id":"mlvu_movie101_110","filename":"movie101_110.mp4","archive":"video_part_5.zip","member":"movie101_110.mp4","bytes":136592934},
{"dataset":"MLVU","video_id":"mlvu_surveil_142","filename":"surveil_142.mp4","archive":"video_part_5.zip","member":"surveil_142.mp4","bytes":72072798},
{"dataset":"MLVU","video_id":"mlvu_needle_53","filename":"needle_53.mp4","archive":"video_part_4.zip","member":"needle_53.mp4","bytes":59685299},
{"dataset":"MLVU","video_id":"mlvu_movie101_84","filename":"movie101_84.mp4","archive":"video_part_6.zip","member":"movie101_84.mp4","bytes":123374449},
{"dataset":"MLVU","video_id":"mlvu_order_253","filename":"order_253.mp4","archive":"video_part_3.zip","member":"order_253.mp4","bytes":165572128},
{"dataset":"MLVU","video_id":"mlvu_en_tv_8","filename":"en_tv_8.mp4","archive":"video_part_6.zip","member":"en_tv_8.mp4","bytes":55645105},
{"dataset":"MLVU","video_id":"mlvu_tomjerry_3","filename":"tomjerry_3.mp4","archive":"video_part_3.zip","member":"tomjerry_3.mp4","bytes":49400760},
{"dataset":"MLVU","video_id":"mlvu_needle_51","filename":"needle_51.mp4","archive":"video_part_5.zip","member":"needle_51.mp4","bytes":109649093},
{"dataset":"MLVU","video_id":"mlvu_count_24","filename":"count_24.mp4","archive":"video_part_5.zip","member":"count_24.mp4","bytes":41598236},
{"dataset":"MLVU","video_id":"mlvu_surveil_28","filename":"surveil_28.mp4","archive":"video_part_3.zip","member":"surveil_28.mp4","bytes":229379791},
{"dataset":"MLVU","video_id":"mlvu_order_157","filename":"order_157.mp4","archive":"video_part_6.zip","member":"order_157.mp4","bytes":81354167},
{"dataset":"MLVU","video_id":"mlvu_movie101_78","filename":"movie101_78.mp4","archive":"video_part_2.zip","member":"movie101_78.mp4","bytes":257667123},
{"dataset":"MLVU","video_id":"mlvu_needle_83","filename":"needle_83.mp4","archive":"video_part_1.zip","member":"needle_83.mp4","bytes":70948155},
{"dataset":"MLVU","video_id":"mlvu_movie101_86","filename":"movie101_86.mp4","archive":"video_part_1.zip","member":"movie101_86.mp4","bytes":141366650},
{"dataset":"MLVU","video_id":"mlvu_movie101_40","filename":"movie101_40.mp4","archive":"video_part_8.zip","member":"movie101_40.mp4","bytes":118372998},
{"dataset":"MLVU","video_id":"mlvu_movie101_63","filename":"movie101_63.mp4","archive":"video_part_8.zip","member":"movie101_63.mp4","bytes":238201847},
{"dataset":"MLVU","video_id":"mlvu_order_25","filename":"order_25.mp4","archive":"video_part_8.zip","member":"order_25.mp4","bytes":98788657},
{"dataset":"MLVU","video_id":"mlvu_surveil_74","filename":"surveil_74.mp4","archive":"video_part_2.zip","member":"surveil_74.mp4","bytes":84029362},
{"dataset":"MLVU","video_id":"mlvu_order_68","filename":"order_68.mp4","archive":"video_part_7.zip","member":"order_68.mp4","bytes":85229854},
{"dataset":"MLVU","video_id":"mlvu_movie101_67","filename":"movie101_67.mp4","archive":"video_part_5.zip","member":"movie101_67.mp4","bytes":187526446},
{"dataset":"MLVU","video_id":"mlvu_ego_25","filename":"ego_25.mp4","archive":"video_part_3.zip","member":"ego_25.mp4","bytes":116835324},
{"dataset":"MLVU","video_id":"mlvu_order_244","filename":"order_244.mp4","archive":"video_part_5.zip","member":"order_244.mp4","bytes":34927631},
{"dataset":"MLVU","video_id":"mlvu_9","filename":"9.mp4","archive":"video_part_3.zip","member":"9.mp4","bytes":43798933},
{"dataset":"MLVU","video_id":"mlvu_ego_56","filename":"ego_56.mp4","archive":"video_part_4.zip","member":"ego_56.mp4","bytes":307030458},
{"dataset":"MLVU","video_id":"mlvu_needle_26","filename":"needle_26.mp4","archive":"video_part_2.zip","member":"needle_26.mp4","bytes":59801515},
{"dataset":"MLVU","video_id":"mlvu_count_5","filename":"count_5.mp4","archive":"video_part_2.zip","member":"count_5.mp4","bytes":51470061},
{"dataset":"MLVU","video_id":"mlvu_en_tv_26","filename":"en_tv_26.mp4","archive":"video_part_4.zip","member":"en_tv_26.mp4","bytes":112396906},
{"dataset":"MLVU","video_id":"mlvu_order_327","filename":"order_327.mp4","archive":"video_part_7.zip","member":"order_327.mp4","bytes":63800127},
{"dataset":"MLVU","video_id":"mlvu_game_31","filename":"game_31.mp4","archive":"video_part_7.zip","member":"game_31.mp4","bytes":85440757},
{"dataset":"MLVU","video_id":"mlvu_order_257","filename":"order_257.mp4","archive":"video_part_1.zip","member":"order_257.mp4","bytes":134972159},
{"dataset":"MLVU","video_id":"mlvu_count_99","filename":"count_99.mp4","archive":"video_part_1.zip","member":"count_99.mp4","bytes":83376134},
{"dataset":"MLVU","video_id":"mlvu_surveil_101","filename":"surveil_101.mp4","archive":"video_part_6.zip","member":"surveil_101.mp4","bytes":105730382},
{"dataset":"MLVU","video_id":"mlvu_surveil_50","filename":"surveil_50.mp4","archive":"video_part_2.zip","member":"surveil_50.mp4","bytes":33429815},
{"dataset":"MLVU","video_id":"mlvu_AWB-11","filename":"AWB-11.mp4","archive":"video_part_5.zip","member":"AWB-11.mp4","bytes":98154725},
{"dataset":"MLVU","video_id":"mlvu_en_tv_31","filename":"en_tv_31.mp4","archive":"video_part_3.zip","member":"en_tv_31.mp4","bytes":240919382},
{"dataset":"MLVU","video_id":"mlvu_ego_36","filename":"ego_36.mp4","archive":"video_part_8.zip","member":"ego_36.mp4","bytes":1091383502},
{"dataset":"MLVU","video_id":"mlvu_tomjerry_10","filename":"tomjerry_10.mp4","archive":"video_part_1.zip","member":"tomjerry_10.mp4","bytes":44782215},
{"dataset":"MLVU","video_id":"mlvu_en_tv_18","filename":"en_tv_18.mp4","archive":"video_part_8.zip","member":"en_tv_18.mp4","bytes":62615315},
{"dataset":"MLVU","video_id":"mlvu_movie101_20","filename":"movie101_20.mp4","archive":"video_part_5.zip","member":"movie101_20.mp4","bytes":149693259},
{"dataset":"MLVU","video_id":"mlvu_en_tv_27","filename":"en_tv_27.mp4","archive":"video_part_8.zip","member":"en_tv_27.mp4","bytes":89289055},
{"dataset":"MLVU","video_id":"mlvu_haimian_2","filename":"haimian_2.mp4","archive":"video_part_6.zip","member":"haimian_2.mp4","bytes":43395114},
{"dataset":"MLVU","video_id":"mlvu_count_172","filename":"count_172.mp4","archive":"video_part_3.zip","member":"count_172.mp4","bytes":62994543},
{"dataset":"MLVU","video_id":"mlvu_surveil_151","filename":"surveil_151.mp4","archive":"video_part_5.zip","member":"surveil_151.mp4","bytes":54684414},
{"dataset":"MLVU","video_id":"mlvu_surveil_186","filename":"surveil_186.mp4","archive":"video_part_5.zip","member":"surveil_186.mp4","bytes":81981405},
{"dataset":"MLVU","video_id":"mlvu_ego_52","filename":"ego_52.mp4","archive":"video_part_2.zip","member":"ego_52.mp4","bytes":268649561},
{"dataset":"MLVU","video_id":"mlvu_ego_2","filename":"ego_2.mp4","archive":"video_part_6.zip","member":"ego_2.mp4","bytes":242916265},
{"dataset":"MLVU","video_id":"mlvu_order_115","filename":"order_115.mp4","archive":"video_part_7.zip","member":"order_115.mp4","bytes":116634040},
{"dataset":"MLVU","video_id":"mlvu_needle_56","filename":"needle_56.mp4","archive":"video_part_1.zip","member":"needle_56.mp4","bytes":73632725},
{"dataset":"MLVU","video_id":"mlvu_needle_31","filename":"needle_31.mp4","archive":"video_part_6.zip","member":"needle_31.mp4","bytes":141038081},
{"dataset":"MLVU","video_id":"mlvu_order_228","filename":"order_228.mp4","archive":"video_part_6.zip","member":"order_228.mp4","bytes":1088856999},
{"dataset":"MLVU","video_id":"mlvu_count_51","filename":"count_51.mp4","archive":"video_part_8.zip","member":"count_51.mp4","bytes":58052088},
{"dataset":"MLVU","video_id":"mlvu_movie101_105","filename":"movie101_105.mp4","archive":"video_part_3.zip","member":"movie101_105.mp4","bytes":300498652},
{"dataset":"MLVU","video_id":"mlvu_count_113","filename":"count_113.mp4","archive":"video_part_8.zip","member":"count_113.mp4","bytes":57109042},
{"dataset":"MLVU","video_id":"mlvu_ego_73","filename":"ego_73.mp4","archive":"video_part_1.zip","member":"ego_73.mp4","bytes":196115508},
{"dataset":"MLVU","video_id":"mlvu_order_345","filename":"order_345.mp4","archive":"video_part_4.zip","member":"order_345.mp4","bytes":66484369},
{"dataset":"MLVU","video_id":"mlvu_ego_29","filename":"ego_29.mp4","archive":"video_part_2.zip","member":"ego_29.mp4","bytes":229652577},
{"dataset":"MLVU","video_id":"mlvu_movie101_74","filename":"movie101_74.mp4","archive":"video_part_7.zip","member":"movie101_74.mp4","bytes":205009630},
{"dataset":"MLVU","video_id":"mlvu_ego_47","filename":"ego_47.mp4","archive":"video_part_5.zip","member":"ego_47.mp4","bytes":4523694831},
{"dataset":"MLVU","video_id":"mlvu_needle_72","filename":"needle_72.mp4","archive":"video_part_7.zip","member":"needle_72.mp4","bytes":75254518},
{"dataset":"MLVU","video_id":"mlvu_order_320","filename":"order_320.mp4","archive":"video_part_3.zip","member":"order_320.mp4","bytes":70944730},
{"dataset":"MLVU","video_id":"mlvu_needle_3","filename":"needle_3.mp4","archive":"video_part_4.zip","member":"needle_3.mp4","bytes":60511723},
{"dataset":"MLVU","video_id":"mlvu_movie101_11","filename":"movie101_11.mp4","archive":"video_part_7.zip","member":"movie101_11.mp4","bytes":66307705},
{"dataset":"MLVU","video_id":"mlvu_ego_41","filename":"ego_41.mp4","archive":"video_part_2.zip","member":"ego_41.mp4","bytes":835932822},
{"dataset":"MLVU","video_id":"mlvu_needle_20","filename":"needle_20.mp4","archive":"video_part_2.zip","member":"needle_20.mp4","bytes":66958225},
{"dataset":"MLVU","video_id":"mlvu_needle_58","filename":"needle_58.mp4","archive":"video_part_4.zip","member":"needle_58.mp4","bytes":70826014},
{"dataset":"MLVU","video_id":"mlvu_ego_42","filename":"ego_42.mp4","archive":"video_part_1.zip","member":"ego_42.mp4","bytes":253033200},
{"dataset":"MLVU","video_id":"mlvu_ego_67","filename":"ego_67.mp4","archive":"video_part_1.zip","member":"ego_67.mp4","bytes":278791630},
{"dataset":"MLVU","video_id":"mlvu_needle_99","filename":"needle_99.mp4","archive":"video_part_5.zip","member":"needle_99.mp4","bytes":59497875},
{"dataset":"MLVU","video_id":"mlvu_needle_52","filename":"needle_52.mp4","archive":"video_part_5.zip","member":"needle_52.mp4","bytes":121038916},
{"dataset":"MLVU","video_id":"mlvu_order_30","filename":"order_30.mp4","archive":"video_part_3.zip","member":"order_30.mp4","bytes":79655660}
]}""")


class DownloadError(RuntimeError):
    pass


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            if urllib.parse.urlparse(newurl).scheme != "https":
                raise DownloadError("Refusing a non-HTTPS redirect.")
            if urllib.parse.urlparse(newurl).hostname != "huggingface.co":
                redirected.remove_header("Authorization")
        return redirected


def hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    token_file = Path(os.environ.get("HF_TOKEN_PATH", str(cache / "token")))
    if token_file.is_file():
        return token_file.read_text().strip()
    if sys.stdin.isatty():
        print("LongVideoBench requires your own HF account with accepted dataset terms.", flush=True)
        return getpass.getpass("Hugging Face read token (hidden, not saved): ").strip()
    return ""


def open_url(opener, url, token, headers=None):
    fields = {"User-Agent": "fixed-video-subset-downloader", "Accept-Encoding": "identity"}
    fields.update(headers or {})
    if token and urllib.parse.urlparse(url).hostname == "huggingface.co":
        fields["Authorization"] = "Bearer " + token
    return opener.open(urllib.request.Request(url, headers=fields), timeout=60)


def public_url(dataset, path):
    return "https://huggingface.co/datasets/" + SOURCES[dataset] + "/resolve/main/" + urllib.parse.quote(path, safe="/") + "?download=true"


def error_note(error):
    if isinstance(error, urllib.error.HTTPError):
        return "HTTP " + str(error.code)
    return type(error).__name__


def catalog(dataset, opener, token):
    url = "https://huggingface.co/api/datasets/" + SOURCES[dataset] + "/tree/main?recursive=true&limit=1000"
    try:
        with open_url(opener, url, token) as response:
            data = json.load(response)
        result = {r["path"]: r["size"] for r in data if r["type"] == "file"}
        for entry in PLAN["catalog"].get(dataset, []):
            if result.get(entry["path"]) != entry["size"]:
                raise DownloadError("Public source changed or is incomplete: " + dataset + "/" + entry["path"])
        return result
    except (urllib.error.URLError, TimeoutError) as error:
        raise DownloadError("Cannot list " + SOURCES[dataset] + " (" + error_note(error) + "). Check network/access.") from None


def download(url, target, size, opener, token):
    if target.is_file():
        if target.stat().st_size == size:
            return target
        raise DownloadError("Existing archive has unexpected size: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise DownloadError("Partial archive exceeds public size: " + str(partial))
        if offset == size:
            partial.replace(target)
            return target
        print(f"DOWNLOAD {target.name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB, attempt {attempt}/5", flush=True)
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with open_url(opener, url, token, headers) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise DownloadError("Server returned an incorrect byte range.")
                elif response.status != 200 or offset:
                    raise DownloadError("Server did not honor resume; partial file preserved.")
                last_report = time.monotonic()
                with partial.open("ab") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        if offset + len(chunk) > size:
                            raise DownloadError("Download exceeds the public file size.")
                        output.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_report >= 30:
                            print(f"  {target.name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB", flush=True)
                            last_report = time.monotonic()
            if offset != size:
                raise DownloadError("Incomplete response; partial file preserved.")
            partial.replace(target)
            return target
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, DownloadError) as error:
            note = error_note(error)
            if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
                raise DownloadError("Download access denied (" + note + "). For LongVideoBench, accept the official terms and use your own HF token.") from None
            if attempt == 5:
                raise DownloadError("Download stopped: " + target.name + " (" + note + "). Rerun to resume.") from None
            print("  Retry: " + note, flush=True)
            time.sleep(attempt * 3)


def video_path(root, row):
    name = row["filename"]
    if PurePosixPath(name).name != name or "\\" in name or name in ("", ".", ".."):
        raise DownloadError("Invalid filename in embedded list.")
    return root / row["dataset"] / "videos" / name


def complete(root, row):
    target = video_path(root, row)
    if not target.exists():
        return False
    if target.is_file() and target.stat().st_size == row["bytes"]:
        return True
    raise DownloadError("Existing video differs from the selected public source: " + str(target))


def write_video(root, row, stream):
    target = video_path(root, row)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with partial.open("wb") as output:
        shutil.copyfileobj(stream, output, 1024 * 1024)
    if partial.stat().st_size != row["bytes"]:
        raise DownloadError("Extracted video size mismatch: " + row["filename"])
    partial.replace(target)
    print("READY " + row["dataset"] + "/" + row["filename"], flush=True)


def extract_zip(root, archive, rows):
    with zipfile.ZipFile(archive) as source:
        for row in rows:
            if complete(root, row):
                continue
            member = source.getinfo(row["member"])
            if member.is_dir() or member.file_size != row["bytes"]:
                raise DownloadError("Archive member differs: " + row["filename"])
            with source.open(member) as stream:
                write_video(root, row, stream)


def extract_tar_parts(root, parts, rows):
    pending = {str(PurePosixPath(r["member"])): r for r in rows if not complete(root, r)}
    # Stream concatenation; no second 161.6 GB merged tar is created.
    process = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|", bufsize=1024 * 1024) as archive:
            for member in archive:
                row = pending.get(str(PurePosixPath(member.name)))
                if row is None:
                    continue
                if not member.isfile() or member.size != row["bytes"]:
                    raise DownloadError("Tar member differs: " + row["filename"])
                with archive.extractfile(member) as stream:
                    write_video(root, row, stream)
                del pending[str(PurePosixPath(member.name))]
                if not pending:
                    break
    finally:
        process.stdout.close()
        code = process.wait()
    if pending or code not in (0, -signal.SIGPIPE):
        raise DownloadError(f"LongVideoBench extraction incomplete: {len(pending)} videos remain.")


def print_plan():
    counts = Counter(r["dataset"] for r in PLAN["rows"])
    print("Frozen 597 questions / 465 unique videos; no resampling.")
    for dataset in SOURCES:
        size = sum(r["bytes"] for r in PLAN["rows"] if r["dataset"] == dataset)
        print(f"  {dataset}: {counts[dataset]} videos, {size / 1e9:.2f} GB; {SOURCES[dataset]}")
    print("Output about 115.1 GB; full archive traffic about 600 GB. Allow >=250 GB free disk.")
    print("LongVideoBench requires accepting the official HF dataset conditions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", default="rebuttal597_videos", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print the frozen plan; no network or file writes.")
    args = parser.parse_args()
    print_plan()
    if args.plan:
        return
    if sys.platform != "linux":
        raise DownloadError("Run this file on Ubuntu Linux.")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent writers in the same target directory.
    import fcntl
    with (root / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError("Another downloader is using this target directory.") from None
        run(root)


def discard_finished_archives(root, groups):
    # Handle interruption after extraction completed but before cache removal.
    for dataset, rows in groups.items():
        cache = root / "_public_archives" / dataset
        if dataset == "LongVideoBench":
            if all(complete(root, r) for r in rows):
                for entry in PLAN["catalog"][dataset]:
                    (cache / entry["path"]).unlink(missing_ok=True)
        else:
            by_archive = defaultdict(list)
            for row in rows:
                by_archive[row["archive"]].append(row)
            for name, selected in by_archive.items():
                if all(complete(root, r) for r in selected):
                    (cache / Path(name).name).unlink(missing_ok=True)


def run(root):
    groups = {ds: [r for r in PLAN["rows"] if r["dataset"] == ds] for ds in SOURCES}
    discard_finished_archives(root, groups)
    active = [ds for ds, rows in groups.items() if not all(complete(root, r) for r in rows)]
    if active:
        token = hf_token() if "LongVideoBench" in active else ""
        opener = urllib.request.build_opener(SafeRedirect())
        catalogs = {ds: catalog(ds, opener, token) for ds in active}
        # Check access before starting any large transfer.
        for dataset in active:
            path = "videos.tar.part.aa" if dataset == "LongVideoBench" else groups[dataset][0]["archive"]
            try:
                with open_url(opener, public_url(dataset, path), token, {"Range": "bytes=0-0"}) as response:
                    response.read(1)
            except (urllib.error.URLError, TimeoutError) as error:
                raise DownloadError("Cannot access " + SOURCES[dataset] + " (" + error_note(error) + "). Check network; accept LongVideoBench terms and supply your own HF token.") from None
        # Download the large joined archive first, to limit peak disk space.
        for dataset in ("LongVideoBench", "LVBench", "Video-MME", "MLVU"):
            if dataset not in active:
                continue
            rows = groups[dataset]
            cache = root / "_public_archives" / dataset
            if dataset == "LongVideoBench":
                names = sorted(x["path"] for x in PLAN["catalog"][dataset])
                paths = [download(public_url(dataset, name), cache / name, catalogs[dataset][name], opener, token) for name in names]
                extract_tar_parts(root, paths, rows)
                for path in paths:
                    path.unlink()
            else:
                by_archive = defaultdict(list)
                for row in rows:
                    by_archive[row["archive"]].append(row)
                for name, selected in sorted(by_archive.items()):
                    if all(complete(root, r) for r in selected):
                        continue
                    if name not in catalogs[dataset]:
                        raise DownloadError("Public archive missing: " + name)
                    path = download(public_url(dataset, name), cache / Path(name).name, catalogs[dataset][name], opener, token)
                    extract_zip(root, path, selected)
                    path.unlink()
    if not all(complete(root, row) for row in PLAN["rows"]):
        raise DownloadError("Some videos remain missing.")
    manifest = {
        "question_count": 597,
        "video_count": 465,
        "files": [dict(row, source_repo=SOURCES[row["dataset"]], path=video_path(root, row).relative_to(root).as_posix()) for row in PLAN["rows"]],
    }
    (root / "video_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE: 465/465 videos. Manifest: " + str(root / "video_manifest.json"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except (DownloadError, OSError, KeyError, ValueError, zipfile.BadZipFile, tarfile.TarError) as error:
        # No traceback with request objects, authenticated URLs, or credentials.
        detail = str(error) if isinstance(error, DownloadError) else type(error).__name__
        print("STOPPED: " + detail, file=sys.stderr)
        sys.exit(1)
