import cv2
import numpy as np

# === CONFIGURATION ===
width, height = 300, 200
fps = 60
output_file = "input_visualizer.mp4"

# === COLORS ===
PINK = (255, 0, 204)
DARK = (30, 30, 30)
WHITE = (255, 255, 255)
BG_COLOR = (0, 255, 0)

# === PARSE INPUT DATA ===
input_text = """
time: 0 steer: 0 accel: 1 brake: 0
time: 1500 steer: 0.1023572534 accel: 1 brake: 0
time: 1600 steer: 0.06426573545 accel: 1 brake: 0
time: 1700 steer: 0.0755302608 accel: 1 brake: 0
time: 1800 steer: 0.1000199988 accel: 1 brake: 0
time: 1900 steer: 0.1289339811 accel: 1 brake: 0
time: 2000 steer: 0.01254066452 accel: 1 brake: 0
time: 2200 steer: -0.2002212554 accel: 1 brake: 0
time: 2300 steer: -0.2099999934 accel: 1 brake: 0
time: 2400 steer: -0.199282825 accel: 1 brake: 0
time: 2500 steer: -0.2245646417 accel: 1 brake: 0
time: 2600 steer: -0.4941856265 accel: 1 brake: 0
time: 2700 steer: -0.8000000119 accel: 1 brake: 0
time: 2800 steer: -0.8100000024 accel: 1 brake: 0
time: 2900 steer: -0.8000000119 accel: 1 brake: 0
time: 3000 steer: -0.81583637 accel: 1 brake: 0
time: 3100 steer: -0.8000000119 accel: 1 brake: 0
time: 3200 steer: -0.01039948687 accel: 1 brake: 0
time: 3300 steer: 0.009999999776 accel: 1 brake: 0
time: 3400 steer: -0.03789544106 accel: 1 brake: 0
time: 3500 steer: 0.04126926512 accel: 1 brake: 0
time: 3600 steer: 0 accel: 1 brake: 0
time: 3700 steer: 0.04915646091 accel: 1 brake: 0
time: 3800 steer: 0 accel: 1 brake: 0
time: 3900 steer: 0.009999999776 accel: 1 brake: 0
time: 4000 steer: -0.04166600853 accel: 1 brake: 0
time: 4100 steer: 0.00254096929 accel: 1 brake: 0
time: 4200 steer: 0.02312204242 accel: 1 brake: 0
time: 4300 steer: -0.01134189777 accel: 1 brake: 0
time: 4400 steer: -0.007960753515 accel: 1 brake: 0
time: 4500 steer: 0.009024322964 accel: 1 brake: 0
time: 4600 steer: 0.03805871308 accel: 1 brake: 0
time: 4700 steer: -0.5 accel: 1 brake: 0
time: 4800 steer: -1 accel: 1 brake: 0
time: 5500 steer: -0.004341868218 accel: 1 brake: 0
time: 5530 steer: -0.005395061802 accel: 1 brake: 0
time: 5540 steer: -0.004341868218 accel: 1 brake: 0
time: 5600 steer: 0 accel: 1 brake: 0
time: 5700 steer: -0.00400000019 accel: 1 brake: 0
time: 5800 steer: -1 accel: 1 brake: 0
time: 5900 steer: -0.001070528757 accel: 1 brake: 0
time: 5910 steer: -0.008490798995 accel: 1 brake: 0
time: 5920 steer: -0.00400000019 accel: 1 brake: 0
time: 6000 steer: 0 accel: 1 brake: 0
time: 6100 steer: -0.00400000019 accel: 1 brake: 0
time: 6200 steer: 0.006195563823 accel: 1 brake: 0
time: 6300 steer: -1 accel: 1 brake: 0
time: 6480 steer: -0.005731681827 accel: 1 brake: 0
time: 6510 steer: 0.01287423354 accel: 1 brake: 0
time: 6520 steer: 0 accel: 1 brake: 0
time: 6600 steer: 0.3000000119 accel: 1 brake: 0
time: 6800 steer: 0.02000854537 accel: 1 brake: 1
time: 6900 steer: 0 accel: 1 brake: 0
time: 7400 steer: 0.7989260554 accel: 1 brake: 0
time: 7500 steer: 1 accel: 1 brake: 0
time: 8840 steer: -1 accel: 1 brake: 0
time: 9800 steer: -0.9907721281 accel: 1 brake: 1
time: 9850 steer: -1 accel: 1 brake: 0
time: 11190 steer: -0.9963331819 accel: 1 brake: 0
time: 11200 steer: -1 accel: 1 brake: 0
time: 11210 steer: -0.9963331819 accel: 1 brake: 0
time: 11300 steer: -0.9942677021 accel: 1 brake: 0
time: 11310 steer: -0.9963331819 accel: 1 brake: 0
time: 11350 steer: -0.9741019607 accel: 1 brake: 0
time: 11360 steer: -0.9972283244 accel: 1 brake: 0
time: 11400 steer: -0.9897933602 accel: 1 brake: 0
time: 11410 steer: -0.9927893877 accel: 1 brake: 0
time: 11540 steer: -0.9746598601 accel: 1 brake: 0
time: 11550 steer: -0.9827125072 accel: 1 brake: 0
time: 11560 steer: -0.9836139083 accel: 1 brake: 0
time: 11570 steer: -0.9928836823 accel: 1 brake: 0
time: 11720 steer: -1 accel: 1 brake: 0
time: 11730 steer: -0.9928836823 accel: 1 brake: 0
time: 11740 steer: -1 accel: 1 brake: 0
time: 11750 steer: -0.9928836823 accel: 1 brake: 0
time: 11900 steer: -1 accel: 1 brake: 0
time: 11910 steer: -0.9895287752 accel: 1 brake: 0
time: 11920 steer: -0.9928836823 accel: 1 brake: 0
time: 11980 steer: -1 accel: 1 brake: 0
time: 12190 steer: -0.01853877306 accel: 1 brake: 0
time: 12200 steer: -0.05103609711 accel: 1 brake: 0
time: 12210 steer: -0.02500869706 accel: 1 brake: 0
time: 12220 steer: -0.01832026057 accel: 1 brake: 0
time: 12230 steer: -0.0115778055 accel: 1 brake: 0
time: 12240 steer: 0.001970579848 accel: 1 brake: 0
time: 12250 steer: -0.03403332084 accel: 1 brake: 0
time: 12260 steer: -0.02566148713 accel: 1 brake: 0
time: 12270 steer: -0.006111939438 accel: 1 brake: 0
time: 12280 steer: -0.01289712079 accel: 1 brake: 0
time: 12290 steer: -0.008143862709 accel: 1 brake: 0
time: 12300 steer: -0.01822412573 accel: 1 brake: 0
time: 12310 steer: -0.01436506025 accel: 1 brake: 0
time: 12320 steer: -0.01822412573 accel: 1 brake: 0
time: 12330 steer: -0.02114016749 accel: 1 brake: 0
time: 12340 steer: -0.01822412573 accel: 1 brake: 0
time: 12350 steer: 0.002813805826 accel: 1 brake: 0
time: 12360 steer: 0.02308054082 accel: 1 brake: 0
time: 12370 steer: -0.02564928122 accel: 1 brake: 0
time: 12380 steer: -0.007299112156 accel: 1 brake: 0
time: 12390 steer: -0.01543626189 accel: 1 brake: 0
time: 12400 steer: -0.009800714441 accel: 1 brake: 0
time: 12410 steer: -0.01828150079 accel: 1 brake: 0
time: 12420 steer: -0.011851863 accel: 1 brake: 0
time: 12430 steer: -0.01495467965 accel: 1 brake: 0
time: 12440 steer: -0.0530808419 accel: 1 brake: 0
time: 12450 steer: -0.02697256207 accel: 1 brake: 0
time: 12460 steer: -0.011851863 accel: 1 brake: 0
time: 12470 steer: -0.01248115487 accel: 1 brake: 0
time: 12480 steer: -0.007064726204 accel: 1 brake: 0
time: 12490 steer: -0.001818596851 accel: 1 brake: 0
time: 12500 steer: 0.00014191214 accel: 1 brake: 0
time: 12510 steer: -0.00104129035 accel: 1 brake: 0
time: 12520 steer: 0.00014191214 accel: 1 brake: 0
time: 12530 steer: -0.001957760658 accel: 1 brake: 0
time: 12540 steer: 0.03428724408 accel: 1 brake: 0
time: 12550 steer: -0.007228002883 accel: 1 brake: 0
time: 12560 steer: -0.001153601799 accel: 1 brake: 0
time: 12570 steer: 0.004219490103 accel: 1 brake: 0
time: 12580 steer: 0.002274547704 accel: 1 brake: 0
time: 12590 steer: 0.04166844487 accel: 1 brake: 0
time: 12600 steer: -0.02918057516 accel: 1 brake: 0
time: 12610 steer: 0.01376995444 accel: 1 brake: 0
time: 12620 steer: -0.004740133882 accel: 1 brake: 0
time: 12630 steer: 0.00014191214 accel: 1 brake: 0
time: 12640 steer: 0.003901486285 accel: 1 brake: 0
time: 12650 steer: 0.0008737449534 accel: 1 brake: 0
time: 12660 steer: -0.02068758383 accel: 1 brake: 0
time: 12670 steer: -0.01432172768 accel: 1 brake: 0
time: 12680 steer: 0.001541489735 accel: 1 brake: 0
time: 12690 steer: -0.02896359004 accel: 1 brake: 0
time: 12700 steer: -0.007865536958 accel: 1 brake: 0
time: 12710 steer: -0.008386485279 accel: 1 brake: 0
time: 12720 steer: -0.03009979427 accel: 1 brake: 0
time: 12740 steer: -0.02373027615 accel: 1 brake: 0
time: 12750 steer: -0.0249052383 accel: 1 brake: 0
time: 12760 steer: -0.02109836042 accel: 1 brake: 0
time: 12770 steer: -0.06576616317 accel: 1 brake: 0
time: 12780 steer: -0.0113626495 accel: 1 brake: 0
time: 12790 steer: -0.03009979427 accel: 1 brake: 0
time: 12800 steer: -0.5 accel: 1 brake: 0
time: 12840 steer: -0.5014517903 accel: 1 brake: 0
time: 12850 steer: -0.5 accel: 1 brake: 0
time: 12900 steer: -0.1000000015 accel: 1 brake: 0
time: 13000 steer: 0 accel: 1 brake: 0
time: 13030 steer: -0.005248267669 accel: 1 brake: 0
time: 13040 steer: 0 accel: 1 brake: 0
time: 13080 steer: 0.002075563185 accel: 1 brake: 0
time: 13090 steer: 0 accel: 1 brake: 0
time: 13110 steer: 0.0009573660791 accel: 1 brake: 0
time: 13120 steer: 0.0007290877402 accel: 1 brake: 0
time: 13130 steer: 0 accel: 1 brake: 0
time: 13190 steer: -0.006115604192 accel: 1 brake: 0
time: 13200 steer: 0.0004910435528 accel: 1 brake: 0
time: 13210 steer: 0 accel: 1 brake: 0
time: 13300 steer: 0.3100000024 accel: 1 brake: 0
time: 13390 steer: 0.3080733716 accel: 1 brake: 0
time: 13400 steer: 0.3093459904 accel: 1 brake: 0
time: 13410 steer: 0.3100000024 accel: 1 brake: 0
time: 13490 steer: 0.3041175604 accel: 1 brake: 0
time: 13500 steer: 0.3249999881 accel: 1 brake: 0
time: 13620 steer: 0.3472675085 accel: 1 brake: 0
time: 13630 steer: 0.325817585 accel: 1 brake: 0
time: 13640 steer: 0.3249999881 accel: 1 brake: 0
time: 13690 steer: 0.3331676722 accel: 1 brake: 0
time: 13700 steer: 0.3249999881 accel: 1 brake: 0
time: 13780 steer: 0.3365775049 accel: 1 brake: 0
time: 13790 steer: 0.3249999881 accel: 1 brake: 0
time: 13820 steer: 0.3175812364 accel: 1 brake: 0
time: 13830 steer: 0.3267404735 accel: 1 brake: 0
time: 13840 steer: 0.3249999881 accel: 1 brake: 0
time: 13890 steer: 0.3282090127 accel: 1 brake: 0
time: 13900 steer: 0.3249999881 accel: 1 brake: 0
time: 13920 steer: 0.3245840073 accel: 1 brake: 0
time: 13930 steer: 0.3249999881 accel: 1 brake: 0
time: 13960 steer: 0.3348352015 accel: 1 brake: 0
time: 13970 steer: 0.3229030669 accel: 1 brake: 0
time: 13980 steer: 0.3249999881 accel: 1 brake: 0
time: 13990 steer: 1 accel: 1 brake: 0
time: 14010 steer: 0.9988399744 accel: 1 brake: 0
time: 14020 steer: 1 accel: 1 brake: 0
time: 14320 steer: 0.9959388971 accel: 1 brake: 0
time: 14330 steer: 1 accel: 1 brake: 0
time: 14600 steer: 0.5400000215 accel: 1 brake: 0
time: 14630 steer: 0.5449800491 accel: 1 brake: 0
time: 14640 steer: 0.5494714379 accel: 1 brake: 0
time: 14650 steer: 0.5400000215 accel: 1 brake: 0
time: 14670 steer: 0.5618015528 accel: 1 brake: 0
time: 14680 steer: 0.5400000215 accel: 1 brake: 0
time: 14690 steer: 0.5496277213 accel: 1 brake: 0
time: 14700 steer: 0.5546519756 accel: 1 brake: 0
time: 14710 steer: 0.9976503849 accel: 1 brake: 0
time: 14720 steer: 1 accel: 1 brake: 0
time: 15460 steer: 0.9928784966 accel: 1 brake: 0
time: 15470 steer: 1 accel: 1 brake: 0
time: 15530 steer: 0.9998623729 accel: 1 brake: 0
time: 15540 steer: 1 accel: 1 brake: 0
time: 15830 steer: 0.9963331819 accel: 1 brake: 0
time: 15840 steer: 1 accel: 1 brake: 0
time: 15880 steer: 1 accel: 1 brake: 1
time: 15900 steer: 0.9988418221 accel: 1 brake: 1
time: 15910 steer: 1 accel: 1 brake: 1
time: 16090 steer: 0.999369204 accel: 1 brake: 1
time: 16100 steer: 1 accel: 1 brake: 1
time: 16350 steer: 0.9944222569 accel: 1 brake: 0
time: 16360 steer: 0.9919999838 accel: 1 brake: 0
time: 16370 steer: 0.9956048131 accel: 1 brake: 0
time: 16380 steer: 0.9919999838 accel: 1 brake: 0
time: 16390 steer: 1 accel: 1 brake: 0
time: 16400 steer: 0.9919999838 accel: 1 brake: 0
time: 16410 steer: 0.9863549471 accel: 1 brake: 0
time: 16420 steer: 1 accel: 1 brake: 0
time: 16430 steer: 0.9919999838 accel: 1 brake: 0
time: 16450 steer: 1 accel: 1 brake: 0
time: 16460 steer: 0.9919999838 accel: 1 brake: 0
time: 16480 steer: 1 accel: 1 brake: 0
time: 16530 steer: 0.9986385703 accel: 1 brake: 0
time: 16540 steer: 1 accel: 1 brake: 0
time: 16730 steer: 0.9989632964 accel: 1 brake: 0
time: 16740 steer: 1 accel: 1 brake: 0
time: 17200 steer: 0.9945619106 accel: 1 brake: 0
time: 17210 steer: 1 accel: 1 brake: 0
time: 17300 steer: 0.9919772744 accel: 1 brake: 0
time: 17310 steer: 1 accel: 1 brake: 0
time: 17340 steer: 0.9994223118 accel: 1 brake: 0
time: 17350 steer: 1 accel: 1 brake: 0
time: 17590 steer: 0.9926972389 accel: 1 brake: 0
time: 17600 steer: 1 accel: 1 brake: 0
time: 17630 steer: 0.9970351458 accel: 1 brake: 0
time: 17640 steer: 1 accel: 1 brake: 0
time: 18080 steer: 0.9982216954 accel: 1 brake: 0
time: 18090 steer: 1 accel: 1 brake: 0
time: 18110 steer: 0.9977285266 accel: 1 brake: 0
time: 18120 steer: 1 accel: 1 brake: 0
time: 18210 steer: 0.9963454008 accel: 1 brake: 0
time: 18220 steer: 1 accel: 1 brake: 0
time: 18250 steer: 0.9986965656 accel: 1 brake: 0
time: 18260 steer: 1 accel: 1 brake: 0
time: 18370 steer: 0.9911969304 accel: 1 brake: 0
time: 18380 steer: 1 accel: 1 brake: 0
time: 18490 steer: 0.9951463342 accel: 1 brake: 0
time: 18500 steer: 1 accel: 1 brake: 0
time: 18510 steer: 0.9915503263 accel: 1 brake: 0
time: 18520 steer: 1 accel: 1 brake: 0
time: 18790 steer: 0.9980000257 accel: 1 brake: 0
time: 18800 steer: -1 accel: 1 brake: 0
time: 18860 steer: -0.9932740331 accel: 1 brake: 0
time: 18870 steer: -1 accel: 1 brake: 0
time: 19000 steer: -0.997638762 accel: 1 brake: 0
time: 19010 steer: -1 accel: 1 brake: 0
time: 19030 steer: -0.9944526553 accel: 1 brake: 0
time: 19040 steer: -1 accel: 1 brake: 0
time: 19200 steer: 1 accel: 1 brake: 0
time: 19220 steer: 0.9983779192 accel: 1 brake: 0
time: 19230 steer: 1 accel: 1 brake: 0
time: 19300 steer: 0.9920752645 accel: 1 brake: 0
time: 19310 steer: 0.98554492 accel: 1 brake: 0
time: 19320 steer: 1 accel: 1 brake: 0
time: 19330 steer: 0.9964485765 accel: 1 brake: 0
time: 19340 steer: 0.991517365 accel: 1 brake: 0
time: 19350 steer: -0.1007501483 accel: 1 brake: 0
time: 19360 steer: -0.09830652922 accel: 1 brake: 0
time: 19370 steer: -0.1000000015 accel: 1 brake: 0
time: 19380 steer: -0.1020529792 accel: 1 brake: 0
time: 19390 steer: -0.1000000015 accel: 1 brake: 0
time: 19400 steer: -0.1173406169 accel: 1 brake: 0
time: 19410 steer: -0.11030671 accel: 1 brake: 0
time: 19420 steer: -0.1047575325 accel: 1 brake: 0
time: 19430 steer: -0.1041880548 accel: 1 brake: 0
time: 19440 steer: -0.1049986333 accel: 1 brake: 0
time: 19450 steer: -0.1092046872 accel: 1 brake: 0
time: 19460 steer: -0.1000000015 accel: 1 brake: 0
time: 19470 steer: -0.1216586828 accel: 1 brake: 0
time: 19480 steer: -0.1084832326 accel: 1 brake: 0
time: 19490 steer: -0.1000000015 accel: 1 brake: 0
time: 19500 steer: -0.1011093482 accel: 1 brake: 0
time: 19510 steer: -0.1070159003 accel: 1 brake: 0
time: 19520 steer: -0.1074883267 accel: 1 brake: 0
time: 19530 steer: -0.1006479114 accel: 1 brake: 0
time: 19540 steer: -0.1038438082 accel: 1 brake: 0
time: 19550 steer: -0.9964107275 accel: 1 brake: 0
time: 19560 steer: -1 accel: 1 brake: 0
time: 19590 steer: -0.9952290654 accel: 1 brake: 0
time: 19600 steer: -0.9968593121 accel: 1 brake: 0
time: 19610 steer: -1 accel: 1 brake: 0
time: 19710 steer: -0.9961000681 accel: 1 brake: 0
time: 19720 steer: -1 accel: 1 brake: 0
time: 19760 steer: -0.9922370315 accel: 1 brake: 0
time: 19770 steer: -1 accel: 1 brake: 0
time: 20150 steer: -0.991756618 accel: 1 brake: 0
time: 20160 steer: -0.9841185212 accel: 1 brake: 0
time: 20170 steer: -1 accel: 1 brake: 0
time: 20320 steer: -0.9909473062 accel: 1 brake: 0
time: 20330 steer: -1 accel: 1 brake: 0
time: 20370 steer: -0.9842811227 accel: 1 brake: 0
time: 20380 steer: -1 accel: 1 brake: 0
time: 21350 steer: 1 accel: 1 brake: 0
time: 21800 steer: 1 accel: 1 brake: 1
time: 21850 steer: 1 accel: 1 brake: 0
time: 22900 steer: 1 accel: 1 brake: 1
time: 22920 steer: 1 accel: 1 brake: 0
time: 24200 steer: 1 accel: 1 brake: 1
time: 24210 steer: 1 accel: 1 brake: 0
time: 25150 steer: 0.9997220039 accel: 1 brake: 0
time: 25160 steer: 1 accel: 1 brake: 0
time: 25860 steer: 0.9817557335 accel: 1 brake: 0
time: 25870 steer: 1 accel: 1 brake: 0
time: 25880 steer: 0.9759474397 accel: 1 brake: 0
time: 25890 steer: 1 accel: 1 brake: 0
time: 26100 steer: 0.9993343949 accel: 1 brake: 0
time: 26110 steer: 0.9749458432 accel: 1 brake: 0
time: 26120 steer: 1 accel: 1 brake: 0
time: 26400 steer: 0 accel: 1 brake: 0
time: 26420 steer: -0.004360789433 accel: 1 brake: 0
time: 26430 steer: 0 accel: 1 brake: 0
time: 26450 steer: 0.03211919963 accel: 1 brake: 0
time: 26460 steer: 0 accel: 1 brake: 0
time: 26500 steer: -0.532715857 accel: 1 brake: 0
time: 26510 steer: -0.5 accel: 1 brake: 0
time: 26530 steer: -0.5037248135 accel: 1 brake: 0
time: 26540 steer: -0.5 accel: 1 brake: 0
time: 26570 steer: -0.4780403972 accel: 1 brake: 0
time: 26580 steer: -0.5 accel: 1 brake: 0
time: 26590 steer: -0.4952455163 accel: 1 brake: 0
time: 26600 steer: 0.0008157603443 accel: 1 brake: 0
time: 26610 steer: 0 accel: 1 brake: 0
time: 26620 steer: 0.0009018220007 accel: 1 brake: 0
time: 26630 steer: 0 accel: 1 brake: 0
time: 26640 steer: 0.009819328785 accel: 1 brake: 0
time: 26650 steer: 0 accel: 1 brake: 0
time: 26680 steer: 0.02119357884 accel: 1 brake: 0
time: 26690 steer: 0 accel: 1 brake: 0
time: 26700 steer: -0.02465223894 accel: 1 brake: 0
time: 26710 steer: 0 accel: 1 brake: 0
time: 26720 steer: 0.0006952118129 accel: 1 brake: 0
time: 26730 steer: 0 accel: 1 brake: 0
time: 26740 steer: 0.01740927994 accel: 1 brake: 0
time: 26750 steer: 0 accel: 1 brake: 0
time: 26760 steer: -0.003781242296 accel: 1 brake: 0
time: 26770 steer: 0 accel: 1 brake: 0
time: 26780 steer: 0.01177312247 accel: 1 brake: 0
time: 26790 steer: 0.06299212575 accel: 1 brake: 0
time: 26800 steer: 0.1428630054 accel: 1 brake: 0
time: 26810 steer: 0.2283464521 accel: 1 brake: 0
time: 26820 steer: 0.3228346407 accel: 1 brake: 0
time: 26830 steer: 0.4641663432 accel: 1 brake: 0
time: 26840 steer: 0.6294252872 accel: 1 brake: 0
time: 26850 steer: 0.6377952695 accel: 1 brake: 0
time: 26860 steer: 0.66590029 accel: 1 brake: 0
time: 26870 steer: 0.6907619238 accel: 1 brake: 0
time: 26880 steer: 0.6507368684 accel: 1 brake: 0
time: 26890 steer: 0.5778325796 accel: 1 brake: 0
time: 26900 steer: 0.4800072014 accel: 1 brake: 0
time: 26910 steer: 0.4173228443 accel: 1 brake: 0
time: 26920 steer: 0.274694562 accel: 1 brake: 0
time: 26930 steer: 0.1712545455 accel: 1 brake: 0
time: 26940 steer: 0.03128659353 accel: 1 brake: 0
time: 26950 steer: -0.002085700631 accel: 1 brake: 0
time: 26960 steer: 0.01605273038 accel: 1 brake: 0
time: 26970 steer: 0.0002932837233 accel: 1 brake: 0
time: 26980 steer: 0 accel: 1 brake: 0
time: 27000 steer: 0.00630787015 accel: 1 brake: 0
time: 27010 steer: 0.06930295378 accel: 1 brake: 0
time: 27020 steer: 0.0276421383 accel: 1 brake: 0
time: 27030 steer: 0.009759818204 accel: 1 brake: 0
time: 27040 steer: -0.0463225171 accel: 1 brake: 0
time: 27050 steer: 0 accel: 1 brake: 0
time: 27070 steer: -0.002878810745 accel: 1 brake: 0
time: 27080 steer: 0 accel: 1 brake: 0
time: 27100 steer: 0.008742026985 accel: 1 brake: 0
time: 27110 steer: 0 accel: 1 brake: 0
time: 27140 steer: 0.03833276778 accel: 1 brake: 0
time: 27150 steer: 0 accel: 1 brake: 0
time: 27170 steer: -0.00123783201 accel: 1 brake: 0
time: 27180 steer: 0.00166173093 accel: 1 brake: 0
time: 27190 steer: 0 accel: 1 brake: 0
time: 27200 steer: 0.00839747116 accel: 1 brake: 0
time: 27210 steer: 0 accel: 1 brake: 0
time: 27220 steer: -0.01044495776 accel: 1 brake: 0
time: 27230 steer: -0.001128878444 accel: 1 brake: 0
time: 27240 steer: -0.01656605676 accel: 1 brake: 0
time: 27250 steer: 0.05751091242 accel: 1 brake: 0
time: 27260 steer: 0 accel: 1 brake: 0
time: 27280 steer: 0.009764401242 accel: 1 brake: 0
time: 27290 steer: 0 accel: 1 brake: 0
time: 27320 steer: 0.06299212575 accel: 1 brake: 0
time: 27330 steer: 0.1417322904 accel: 1 brake: 0
time: 27340 steer: 0.2204849571 accel: 1 brake: 0
time: 27350 steer: 0.3700787425 accel: 1 brake: 0
time: 27360 steer: 0.4368084967 accel: 1 brake: 0
time: 27370 steer: 0.4781219661 accel: 1 brake: 0
time: 27380 steer: 0.488188982 accel: 1 brake: 0
time: 27390 steer: 0.5073012114 accel: 1 brake: 0
time: 27400 steer: 0.4936807752 accel: 1 brake: 0
time: 27410 steer: 0.3779527545 accel: 1 brake: 0
time: 27420 steer: 0.3087065816 accel: 1 brake: 0
time: 27430 steer: 0.2992126048 accel: 1 brake: 0
time: 27440 steer: 0.2989014387 accel: 1 brake: 0
time: 27450 steer: 0.236220479 accel: 1 brake: 0
time: 27460 steer: 0.9805545211 accel: 1 brake: 0
time: 27470 steer: 0.9909186363 accel: 1 brake: 0
time: 27480 steer: 1 accel: 1 brake: 0
time: 27490 steer: 0.9968605638 accel: 1 brake: 0
time: 27500 steer: 1 accel: 1 brake: 0
time: 27520 steer: 0.9531254172 accel: 1 brake: 0
time: 27530 steer: 0.3566532731 accel: 1 brake: 0
time: 27540 steer: 0.3823067546 accel: 1 brake: 0
time: 27550 steer: 0.4655091465 accel: 1 brake: 0
time: 27560 steer: 0.5701834559 accel: 1 brake: 0
time: 27570 steer: 0.6104829311 accel: 1 brake: 0
time: 27580 steer: 0.6122590899 accel: 1 brake: 0
time: 27590 steer: 0.6274276972 accel: 1 brake: 0
time: 27600 steer: 0.8144465685 accel: 1 brake: 0
time: 27610 steer: 0.8602145314 accel: 1 brake: 0
time: 27620 steer: 0.814314723 accel: 1 brake: 0
time: 27630 steer: 0.7818042636 accel: 1 brake: 0
time: 27640 steer: 0.8155070543 accel: 1 brake: 0
time: 27650 steer: 0.827665329 accel: 1 brake: 0
time: 27660 steer: 0.8000000119 accel: 1 brake: 0
time: 27670 steer: 0.8283721805 accel: 1 brake: 0
time: 27680 steer: 0.8451285958 accel: 1 brake: 0
time: 27690 steer: 0.8062990308 accel: 1 brake: 0
time: 27700 steer: 0.832542479 accel: 1 brake: 0
time: 27710 steer: 0.7861119509 accel: 1 brake: 0
time: 27720 steer: 0.7664009929 accel: 1 brake: 0
time: 27730 steer: 0.8061046004 accel: 1 brake: 0
time: 27740 steer: 0.7790759206 accel: 1 brake: 0
time: 27750 steer: 0.8127964735 accel: 1 brake: 0
time: 27760 steer: 0.8048607111 accel: 1 brake: 0
time: 27770 steer: 0.7902429104 accel: 1 brake: 0
time: 27780 steer: 0.8000000119 accel: 1 brake: 0
time: 27790 steer: 0.7424695492 accel: 1 brake: 0
time: 27800 steer: 0.788887918 accel: 1 brake: 0
time: 27810 steer: 0.7943034172 accel: 1 brake: 0
time: 27820 steer: 0.8240504265 accel: 1 brake: 0
time: 27830 steer: 0.8000000119 accel: 1 brake: 0
time: 27840 steer: 0.8165871501 accel: 1 brake: 0
time: 27850 steer: 0.794455409 accel: 1 brake: 0
time: 27860 steer: 0.8374233842 accel: 1 brake: 0
time: 27870 steer: 0.7975771427 accel: 1 brake: 0
time: 27880 steer: 0.828884244 accel: 1 brake: 0
time: 27890 steer: 0.8000000119 accel: 1 brake: 0
time: 27900 steer: 0.7042790055 accel: 1 brake: 0
time: 27910 steer: 0.764053762 accel: 1 brake: 0
time: 27920 steer: 0.7968111038 accel: 1 brake: 0
time: 27930 steer: 0.8000000119 accel: 1 brake: 0
time: 27940 steer: 0.796756804 accel: 1 brake: 0
time: 27950 steer: 0.796163559 accel: 1 brake: 0
time: 27960 steer: 0.871080637 accel: 1 brake: 0
time: 27970 steer: 0.7977358699 accel: 1 brake: 0
time: 27980 steer: 0.8025256991 accel: 1 brake: 0
time: 27990 steer: 0.8000000119 accel: 1 brake: 0
time: 28010 steer: 0.8037641644 accel: 1 brake: 0
time: 28020 steer: 0.8390625119 accel: 1 brake: 0
time: 28030 steer: 0.7744935751 accel: 1 brake: 0
time: 28040 steer: 0.7820175886 accel: 1 brake: 0
time: 28050 steer: 0.8303369284 accel: 1 brake: 0
time: 28060 steer: 0.8000000119 accel: 1 brake: 0
time: 28070 steer: 0.787379086 accel: 1 brake: 0
time: 28080 steer: 0.8255211115 accel: 1 brake: 0
time: 28090 steer: 0.7734641433 accel: 1 brake: 0
time: 28100 steer: 0.7923713326 accel: 1 brake: 0
time: 28110 steer: 0.8401678801 accel: 1 brake: 0
time: 28120 steer: 0.7621690035 accel: 1 brake: 0
time: 28130 steer: 0.8000000119 accel: 1 brake: 0
time: 28140 steer: 0.8134871125 accel: 1 brake: 0
time: 28150 steer: 1 accel: 1 brake: 0
time: 28210 steer: 0.9989516735 accel: 1 brake: 0
time: 28220 steer: 1 accel: 1 brake: 0
time: 28260 steer: 0.9907806516 accel: 1 brake: 0
time: 28270 steer: 1 accel: 1 brake: 0
time: 28280 steer: 0.9934315085 accel: 1 brake: 0
time: 28290 steer: 1 accel: 1 brake: 0
time: 28370 steer: 0.9864378572 accel: 1 brake: 0
time: 28380 steer: 1 accel: 1 brake: 0
time: 28400 steer: 0.9895614386 accel: 1 brake: 0
time: 28410 steer: 0.9910498261 accel: 1 brake: 0
time: 28420 steer: 1 accel: 1 brake: 0
time: 28430 steer: 0.9687862992 accel: 1 brake: 0
time: 28440 steer: 1 accel: 1 brake: 0
time: 28450 steer: -0.9636133909 accel: 1 brake: 0
time: 28460 steer: -1 accel: 1 brake: 0
time: 28520 steer: -0.9975789785 accel: 1 brake: 0
time: 28530 steer: -1 accel: 1 brake: 0
time: 28860 steer: -0.9879558682 accel: 1 brake: 0
time: 28870 steer: -1 accel: 1 brake: 0
time: 28880 steer: -0.9921930432 accel: 1 brake: 0
time: 28890 steer: -1 accel: 1 brake: 0
time: 28950 steer: -0.9895461798 accel: 1 brake: 0
time: 28960 steer: -1 accel: 1 brake: 0
time: 29050 steer: -0.9981167316 accel: 1 brake: 0
time: 29060 steer: -1 accel: 1 brake: 0
time: 29100 steer: -0.9816788435 accel: 1 brake: 0
time: 29110 steer: -1 accel: 1 brake: 0
time: 29150 steer: -0.9977632761 accel: 1 brake: 0
time: 29160 steer: -0.984816432 accel: 1 brake: 0
time: 29170 steer: -1 accel: 1 brake: 0
time: 29210 steer: -0.998703897 accel: 1 brake: 0
time: 29220 steer: -1 accel: 1 brake: 0
time: 29270 steer: -0.9973940253 accel: 1 brake: 0
time: 29280 steer: -1 accel: 1 brake: 0
time: 29330 steer: -0.998453021 accel: 1 brake: 0
time: 29340 steer: -1 accel: 1 brake: 0
time: 29400 steer: -0.9987862706 accel: 1 brake: 0
time: 29410 steer: -1 accel: 1 brake: 0
time: 29450 steer: -0.9980331063 accel: 1 brake: 0
time: 29460 steer: -1 accel: 1 brake: 0
time: 29530 steer: -0.9978768229 accel: 1 brake: 0
time: 29540 steer: -0.9998483062 accel: 1 brake: 0
time: 29550 steer: -1 accel: 1 brake: 0
time: 29580 steer: -0.9970174432 accel: 1 brake: 0
time: 29590 steer: -1 accel: 1 brake: 0
time: 29700 steer: -0.9914618134 accel: 1 brake: 0
time: 29710 steer: -1 accel: 1 brake: 0
time: 29780 steer: -0.9968562722 accel: 1 brake: 0
time: 29790 steer: -1 accel: 1 brake: 0
time: 29870 steer: -0.9999667406 accel: 1 brake: 0
time: 29880 steer: -1 accel: 1 brake: 0
time: 29890 steer: -0.9990560412 accel: 1 brake: 0
time: 29900 steer: -1 accel: 1 brake: 0
time: 30190 steer: -0.9977577925 accel: 1 brake: 0
time: 30200 steer: -1 accel: 1 brake: 1
time: 30530 steer: -0.9902380705 accel: 1 brake: 1
time: 30540 steer: -1 accel: 1 brake: 1
time: 30600 steer: -1 accel: 1 brake: 0
time: 30820 steer: -1 accel: 1 brake: 1
time: 30860 steer: -0.9941374063 accel: 1 brake: 1
time: 30870 steer: -1 accel: 1 brake: 1
time: 30950 steer: -0.9987154603 accel: 1 brake: 1
time: 30960 steer: -1 accel: 1 brake: 1
time: 31000 steer: -1 accel: 1 brake: 0
time: 31100 steer: -0.9944010973 accel: 1 brake: 0
time: 31110 steer: -1 accel: 1 brake: 0
time: 31190 steer: -0.9956215024 accel: 1 brake: 0
time: 31200 steer: -0.9912347794 accel: 1 brake: 0
time: 31210 steer: -1 accel: 1 brake: 0
time: 31240 steer: -0.9925833941 accel: 1 brake: 0
time: 31250 steer: -0.9905956984 accel: 1 brake: 0
time: 31260 steer: -1 accel: 1 brake: 0
time: 31360 steer: -0.9993728399 accel: 1 brake: 0
time: 31370 steer: -1 accel: 1 brake: 0
time: 31710 steer: -0.9955549836 accel: 1 brake: 0
time: 31720 steer: -0.9985513091 accel: 1 brake: 0
time: 31730 steer: -1 accel: 1 brake: 0
time: 31820 steer: -0.9973891377 accel: 1 brake: 0
time: 31830 steer: -1 accel: 1 brake: 0
time: 31870 steer: -0.993440032 accel: 1 brake: 0
time: 31880 steer: -1 accel: 1 brake: 0
time: 31900 steer: -0.996769011 accel: 1 brake: 0
time: 31910 steer: -1 accel: 1 brake: 0
time: 31940 steer: -0.9999533296 accel: 1 brake: 0
time: 31950 steer: -1 accel: 1 brake: 0
time: 32050 steer: -0.9952961802 accel: 1 brake: 0
time: 32060 steer: -1 accel: 1 brake: 0
time: 32120 steer: -0.9965630174 accel: 1 brake: 0
time: 32130 steer: -1 accel: 1 brake: 0
time: 32320 steer: -0.9956480861 accel: 1 brake: 0
time: 32330 steer: -1 accel: 1 brake: 0
time: 32370 steer: -0.982301712 accel: 1 brake: 0
time: 32380 steer: -1 accel: 1 brake: 0
time: 32460 steer: -0.9914072156 accel: 1 brake: 0
time: 32470 steer: -1 accel: 1 brake: 0
time: 32480 steer: -0.9951133728 accel: 1 brake: 0
time: 32490 steer: -0.9664967656 accel: 1 brake: 0
time: 32500 steer: -1 accel: 1 brake: 0
time: 32520 steer: -0.9925379157 accel: 1 brake: 0
time: 32530 steer: -0.9715237617 accel: 1 brake: 0
time: 32540 steer: -1 accel: 1 brake: 0
time: 32570 steer: -0.9790249467 accel: 1 brake: 0
time: 32580 steer: -0.9438337088 accel: 1 brake: 0
time: 32590 steer: -0.9913116693 accel: 1 brake: 0
time: 32600 steer: -1 accel: 1 brake: 0
time: 32610 steer: -0.9610131979 accel: 1 brake: 0
time: 32620 steer: -0.9816592932 accel: 1 brake: 0
time: 32630 steer: -1 accel: 1 brake: 0
time: 32640 steer: -0.9823960066 accel: 1 brake: 0
time: 32650 steer: -1 accel: 1 brake: 0
time: 32670 steer: -0.9977431297 accel: 1 brake: 0
time: 32680 steer: -0.9931272268 accel: 1 brake: 0
time: 32690 steer: -0.9547972083 accel: 1 brake: 0
time: 32700 steer: -0.9709112644 accel: 1 brake: 0
time: 32710 steer: -0.9930054545 accel: 1 brake: 0
time: 32720 steer: -0.9731962085 accel: 1 brake: 0
time: 32730 steer: -1 accel: 1 brake: 0
time: 32750 steer: -0.9890878201 accel: 1 brake: 0
time: 32760 steer: -0.9881136417 accel: 1 brake: 0
time: 32770 steer: -0.9923450351 accel: 1 brake: 0
time: 32780 steer: -1 accel: 1 brake: 0
time: 32820 steer: -0.9645169973 accel: 1 brake: 0
time: 32830 steer: -0.9433540106 accel: 1 brake: 0
time: 32840 steer: -1 accel: 1 brake: 0
time: 32850 steer: -0.9535755515 accel: 1 brake: 0
time: 32860 steer: -0.9924365878 accel: 1 brake: 0
time: 32870 steer: -0.982390821 accel: 1 brake: 0
time: 32880 steer: -0.9501376152 accel: 1 brake: 0
time: 32890 steer: -1 accel: 1 brake: 0
time: 32900 steer: -0.9858107567 accel: 1 brake: 0
time: 32910 steer: -1 accel: 1 brake: 0
time: 32930 steer: -0.9693416953 accel: 1 brake: 0
time: 32940 steer: -1 accel: 1 brake: 0
time: 32950 steer: -0.9889477491 accel: 1 brake: 0
time: 32960 steer: -0.9926569462 accel: 1 brake: 0
time: 32970 steer: -1 accel: 1 brake: 0
time: 32990 steer: -0.9645546079 accel: 1 brake: 0
time: 33000 steer: -0.429048121 accel: 1 brake: 0
time: 33010 steer: -0.400000006 accel: 1 brake: 0
time: 33020 steer: -0.4086034894 accel: 1 brake: 0
time: 33030 steer: -0.3917892873 accel: 1 brake: 0
time: 33040 steer: -0.3764726818 accel: 1 brake: 0
time: 33050 steer: -0.3634937704 accel: 1 brake: 0
time: 33060 steer: -0.4370448887 accel: 1 brake: 0
time: 33070 steer: -0.4083416164 accel: 1 brake: 0
time: 33080 steer: -0.3955601752 accel: 1 brake: 0
time: 33090 steer: -0.4024680257 accel: 1 brake: 0
time: 33100 steer: -0.3996032476 accel: 1 brake: 0
time: 33110 steer: -0.4173696041 accel: 1 brake: 0
time: 33120 steer: -0.400000006 accel: 1 brake: 0
time: 33130 steer: -0.4471376538 accel: 1 brake: 0
time: 33140 steer: -0.4257335067 accel: 1 brake: 0
time: 33150 steer: -0.4103518724 accel: 1 brake: 0
time: 33160 steer: -0.400000006 accel: 1 brake: 0
time: 33170 steer: -1 accel: 1 brake: 0
time: 33180 steer: -0.9746089578 accel: 1 brake: 0
time: 33190 steer: -0.9640430808 accel: 1 brake: 0
time: 33200 steer: -0.9725077152 accel: 1 brake: 0
time: 33210 steer: -1 accel: 1 brake: 0
time: 33690 steer: -0.8000000119 accel: 1 brake: 0
time: 33700 steer: 1 accel: 1 brake: 0
time: 33720 steer: 0.9716678262 accel: 1 brake: 0
time: 33730 steer: 1 accel: 1 brake: 0
time: 33740 steer: 0.9951201081 accel: 1 brake: 0
time: 33750 steer: 1 accel: 1 brake: 0
time: 34500 steer: -1 accel: 1 brake: 0
time: 34530 steer: -0.9999529719 accel: 1 brake: 0
time: 34540 steer: -1 accel: 1 brake: 0
time: 34590 steer: -1 accel: 1 brake: 1
time: 34630 steer: -1 accel: 1 brake: 0
time: 35280 steer: -0.9982085824 accel: 1 brake: 0
time: 35290 steer: -1 accel: 1 brake: 0
time: 35590 steer: -0.9969316721 accel: 1 brake: 0
time: 35600 steer: -1 accel: 1 brake: 0
time: 35900 steer: -0.9974065423 accel: 1 brake: 0
time: 35910 steer: -1 accel: 1 brake: 0
time: 35940 steer: -1 accel: 1 brake: 1
time: 35950 steer: -1 accel: 1 brake: 0
time: 36300 steer: -1 accel: 1 brake: 1
time: 36320 steer: -1 accel: 1 brake: 0
time: 36520 steer: -0.9984405041 accel: 1 brake: 0
time: 36530 steer: -1 accel: 1 brake: 0
time: 37000 steer: -1 accel: 1 brake: 1
time: 37010 steer: -1 accel: 1 brake: 0
time: 37100 steer: -1 accel: 1 brake: 1
time: 37110 steer: -1 accel: 1 brake: 0
time: 37250 steer: -1 accel: 1 brake: 1
time: 37260 steer: -1 accel: 1 brake: 0
time: 37350 steer: -1 accel: 1 brake: 1
time: 37360 steer: -1 accel: 1 brake: 0
time: 37440 steer: -0.9854872227 accel: 1 brake: 0
time: 37450 steer: -1 accel: 1 brake: 1
time: 37460 steer: -0.9999981523 accel: 1 brake: 0
time: 37470 steer: -1 accel: 1 brake: 0
time: 37640 steer: -0.9912418127 accel: 1 brake: 0
time: 37650 steer: -1 accel: 1 brake: 0
time: 37960 steer: -0.993555069 accel: 1 brake: 0
time: 37970 steer: -1 accel: 1 brake: 0
time: 38260 steer: -0.9994378686 accel: 1 brake: 0
time: 38270 steer: -1 accel: 1 brake: 0
time: 38390 steer: -0.9895107746 accel: 1 brake: 0
time: 38400 steer: -1 accel: 1 brake: 0
time: 38770 steer: -0.9959904552 accel: 1 brake: 0
time: 38780 steer: -1 accel: 1 brake: 0
time: 38950 steer: -0.9932914376 accel: 1 brake: 0
time: 38960 steer: -1 accel: 1 brake: 0
time: 39280 steer: -0.9976286888 accel: 1 brake: 0
time: 39290 steer: -1 accel: 1 brake: 0
time: 39390 steer: -0.9854829907 accel: 1 brake: 0
time: 39400 steer: -1 accel: 1 brake: 0
time: 39410 steer: -0.9853602648 accel: 1 brake: 0
time: 39420 steer: -1 accel: 1 brake: 0
time: 39440 steer: -0.9900223017 accel: 1 brake: 0
time: 39450 steer: -0.9914346933 accel: 1 brake: 0
time: 39460 steer: -1 accel: 1 brake: 0
time: 39480 steer: -0.9692989588 accel: 1 brake: 0
time: 39490 steer: -0.9936137199 accel: 1 brake: 0
time: 39500 steer: -1 accel: 1 brake: 0
time: 39530 steer: -0.9884023666 accel: 1 brake: 0
time: 39540 steer: -1 accel: 1 brake: 0
time: 39550 steer: -0.9680446982 accel: 1 brake: 0
time: 39560 steer: -1 accel: 1 brake: 0
time: 39570 steer: -0.9971801043 accel: 1 brake: 0
time: 39580 steer: -1 accel: 1 brake: 0
time: 39590 steer: -0.9889211655 accel: 1 brake: 0
time: 39600 steer: -0.9868831635 accel: 1 brake: 0
time: 39610 steer: -0.9811255336 accel: 1 brake: 0
time: 39620 steer: -1 accel: 1 brake: 0
time: 39640 steer: -0.9523502588 accel: 1 brake: 0
time: 39650 steer: -1 accel: 1 brake: 0
time: 39760 steer: -0.9818213582 accel: 1 brake: 0
time: 39770 steer: -0.9816205502 accel: 1 brake: 0
time: 39780 steer: -0.977185607 accel: 1 brake: 0
time: 39790 steer: -0.9940336347 accel: 1 brake: 0
time: 39800 steer: -1 accel: 1 brake: 0
time: 39820 steer: -0.9849128723 accel: 1 brake: 0
time: 39830 steer: -1 accel: 1 brake: 0
time: 39840 steer: -0.9884902239 accel: 1 brake: 0
time: 39850 steer: -1 accel: 1 brake: 0
time: 39960 steer: -0.9991521835 accel: 1 brake: 0
time: 39970 steer: -1 accel: 1 brake: 0
time: 40000 steer: 0.008835108951 accel: 1 brake: 0
time: 40010 steer: 0.02081179246 accel: 1 brake: 0
time: 40020 steer: -0.001352580264 accel: 1 brake: 0
time: 40030 steer: 0.02401684411 accel: 1 brake: 0
time: 40040 steer: 0 accel: 1 brake: 0
time: 40050 steer: 0.03783134744 accel: 1 brake: 0
time: 40060 steer: -0.02454176545 accel: 1 brake: 0
time: 40070 steer: -0.00626728218 accel: 1 brake: 0
time: 40080 steer: 0.01542954519 accel: 1 brake: 0
time: 40090 steer: -0.004654685035 accel: 1 brake: 0
time: 40100 steer: -0.003183690831 accel: 1 brake: 0
time: 40110 steer: 0.01231544092 accel: 1 brake: 0
time: 40120 steer: 0.004424573854 accel: 1 brake: 0
time: 40130 steer: 0.01358867157 accel: 1 brake: 0
time: 40140 steer: 0.005554977804 accel: 1 brake: 0
time: 40150 steer: 0 accel: 1 brake: 0
time: 40160 steer: 0.01312784106 accel: 1 brake: 0
time: 40170 steer: -0.0008557382971 accel: 1 brake: 0
time: 40180 steer: -0.0102121029 accel: 1 brake: 0
time: 40190 steer: 1 accel: 1 brake: 0
time: 40240 steer: 0.9814917445 accel: 1 brake: 0
time: 40250 steer: 0.9646705389 accel: 1 brake: 0
time: 40260 steer: 1 accel: 1 brake: 0
time: 40330 steer: 0.9881234169 accel: 1 brake: 0
time: 40340 steer: 0.9939713478 accel: 1 brake: 0
time: 40350 steer: 0.987757802 accel: 1 brake: 0
time: 40360 steer: 1 accel: 1 brake: 0
time: 40370 steer: 0.9905435443 accel: 1 brake: 0
time: 40380 steer: 1 accel: 1 brake: 0
time: 40390 steer: 0.9768449664 accel: 1 brake: 0
time: 40400 steer: 0.9699661136 accel: 1 brake: 0
time: 40410 steer: 0.9821063876 accel: 1 brake: 0
time: 40420 steer: 0.9804370403 accel: 1 brake: 0
time: 40430 steer: 1 accel: 1 brake: 0
time: 40450 steer: 0.02267708629 accel: 1 brake: 0
time: 40460 steer: 0.03210303187 accel: 1 brake: 0
time: 40470 steer: 0.01819452643 accel: 1 brake: 0
time: 40480 steer: 0 accel: 1 brake: 0
time: 40490 steer: 0.04008972272 accel: 1 brake: 0
time: 40500 steer: 0.00627277419 accel: 1 brake: 0
time: 40510 steer: 0.0395818986 accel: 1 brake: 0
time: 40520 steer: 0.01194433495 accel: 1 brake: 0
time: 40530 steer: 0.03188939765 accel: 1 brake: 0
time: 40540 steer: 0.01596301049 accel: 1 brake: 0
time: 40550 steer: 0.008969390765 accel: 1 brake: 0
time: 40560 steer: 0.01025971025 accel: 1 brake: 0
time: 40570 steer: 0 accel: 1 brake: 0
time: 40580 steer: -0.008750265464 accel: 1 brake: 0
time: 40590 steer: 0.01572008431 accel: 1 brake: 0
time: 40600 steer: 0.01306558214 accel: 1 brake: 0
time: 40610 steer: -0.01306375302 accel: 1 brake: 0
time: 40620 steer: 0.01439069584 accel: 1 brake: 0
time: 40630 steer: 0 accel: 1 brake: 0
time: 40640 steer: 0.04389904439 accel: 1 brake: 0
time: 40650 steer: 0.005579393357 accel: 1 brake: 0
time: 40660 steer: 0.00116947107 accel: 1 brake: 0
time: 40670 steer: 0.002165594138 accel: 1 brake: 0
time: 40680 steer: 0.01306070015 accel: 1 brake: 0
time: 40690 steer: 0.006042664871 accel: 1 brake: 0
time: 40700 steer: 0.02367137372 accel: 1 brake: 0
time: 40710 steer: 0.01129856333 accel: 1 brake: 0
time: 40720 steer: -0.006618243642 accel: 1 brake: 0
time: 40730 steer: 0.004682758823 accel: 1 brake: 0
time: 40740 steer: 0.03107943945 accel: 1 brake: 0
time: 40750 steer: 0.01065950654 accel: 1 brake: 0
time: 40760 steer: 0.03863460571 accel: 1 brake: 0
time: 40770 steer: 0.01766350307 accel: 1 brake: 0
time: 40780 steer: -0.01589709148 accel: 1 brake: 0
time: 40790 steer: 0 accel: 1 brake: 0
time: 40800 steer: 0.0009564515203 accel: 1 brake: 0
time: 40810 steer: -0.0002752766013 accel: 1 brake: 0
time: 40820 steer: -0.01404217537 accel: 1 brake: 0
time: 40830 steer: -0.001810967922 accel: 1 brake: 0
time: 40840 steer: 0.00555986166 accel: 1 brake: 0
time: 40850 steer: 0 accel: 1 brake: 0
time: 40860 steer: 0.01810357906 accel: 1 brake: 0
time: 40870 steer: 0.01289162785 accel: 1 brake: 0
time: 40880 steer: -0.00104922615 accel: 1 brake: 0
time: 40890 steer: -0.008180790581 accel: 1 brake: 0
time: 40900 steer: 0.0160930194 accel: 1 brake: 0
time: 40910 steer: 0.01380779315 accel: 1 brake: 0
time: 40920 steer: 0.01053498872 accel: 1 brake: 0
time: 40930 steer: -0.0209283717 accel: 1 brake: 0
time: 40940 steer: -3.540143371e-05 accel: 1 brake: 0
time: 40950 steer: -0.01842768677 accel: 1 brake: 0
time: 40960 steer: 0.0113498345 accel: 1 brake: 0
time: 40970 steer: 0 accel: 1 brake: 0
time: 40980 steer: -0.01301675476 accel: 1 brake: 0
time: 40990 steer: 0.01513535157 accel: 1 brake: 0
time: 41000 steer: 0.1000000015 accel: 1 brake: 0
time: 41040 steer: 0.08518815041 accel: 1 brake: 0
time: 41050 steer: 0.1000000015 accel: 1 brake: 0
time: 41080 steer: 0.08789697289 accel: 1 brake: 0
time: 41090 steer: 0.1065804064 accel: 1 brake: 0
time: 41100 steer: 0.200000003 accel: 1 brake: 0
time: 41130 steer: 0.2151121497 accel: 1 brake: 0
time: 41140 steer: 0.200000003 accel: 1 brake: 0
time: 41200 steer: 0 accel: 1 brake: 0
time: 41260 steer: -0.0124570448 accel: 1 brake: 0
time: 41270 steer: 0 accel: 1 brake: 0
time: 41300 steer: 0.1000000015 accel: 1 brake: 0
time: 41330 steer: 0.08537980169 accel: 1 brake: 0
time: 41340 steer: 0.1000000015 accel: 1 brake: 0
time: 41400 steer: -1 accel: 1 brake: 0
time: 41490 steer: -0.982429266 accel: 1 brake: 0
time: 41500 steer: -1 accel: 1 brake: 0
time: 41760 steer: -0.9911929965 accel: 1 brake: 0
time: 41770 steer: -1 accel: 1 brake: 0
time: 41940 steer: -0.9888198376 accel: 1 brake: 0
time: 41950 steer: -1 accel: 1 brake: 0
time: 42230 steer: 0 accel: 1 brake: 0
time: 42300 steer: -1 accel: 1 brake: 0
time: 42390 steer: -0.982003212 accel: 1 brake: 0
time: 42400 steer: -1 accel: 1 brake: 0
time: 42510 steer: -0.9872938991 accel: 1 brake: 0
time: 42520 steer: -1 accel: 1 brake: 0
time: 42590 steer: -0.9934000969 accel: 1 brake: 0
time: 42600 steer: -1 accel: 1 brake: 0
time: 42680 steer: -0.9890010953 accel: 1 brake: 0
time: 42690 steer: -1 accel: 1 brake: 0
time: 42800 steer: -0.9819360971 accel: 1 brake: 0
time: 42810 steer: -1 accel: 1 brake: 0
time: 42840 steer: -0.9857020974 accel: 1 brake: 0
time: 42850 steer: -1 accel: 1 brake: 0
time: 42860 steer: -0.9868349433 accel: 1 brake: 0
time: 42870 steer: -0.9948094487 accel: 1 brake: 0
time: 42880 steer: -1 accel: 1 brake: 0
time: 42900 steer: 0.200000003 accel: 1 brake: 0
time: 42950 steer: 0.19003205 accel: 1 brake: 0
time: 42960 steer: 0.200000003 accel: 1 brake: 0
time: 42980 steer: 0.2336106598 accel: 1 brake: 0
time: 42990 steer: 0.2047352493 accel: 1 brake: 0
time: 43000 steer: 0.200000003 accel: 1 brake: 0
time: 43030 steer: 0.2070781589 accel: 1 brake: 0
time: 43040 steer: 0.200000003 accel: 1 brake: 0
time: 43050 steer: 0.2155992389 accel: 1 brake: 0
time: 43060 steer: 0.200000003 accel: 1 brake: 0
time: 43070 steer: 0.2041132897 accel: 1 brake: 0
time: 43080 steer: 0.200000003 accel: 1 brake: 0
time: 43090 steer: 0.2199426293 accel: 1 brake: 0
time: 43100 steer: 0.9820520878 accel: 1 brake: 0
time: 43110 steer: 0.9954176545 accel: 1 brake: 0
time: 43120 steer: 1 accel: 1 brake: 0
time: 43200 steer: 0.981185317 accel: 1 brake: 0
time: 43210 steer: 0.9920310378 accel: 1 brake: 0
time: 43220 steer: 1 accel: 1 brake: 0
time: 43240 steer: 0.9961320758 accel: 1 brake: 0
time: 43250 steer: 1 accel: 1 brake: 0
time: 43310 steer: 0.9961247444 accel: 1 brake: 0
time: 43320 steer: 1 accel: 1 brake: 0
time: 43330 steer: 0.9971074462 accel: 1 brake: 0
time: 43340 steer: 1 accel: 1 brake: 0
time: 43450 steer: 0.9918777943 accel: 1 brake: 0
time: 43460 steer: 1 accel: 1 brake: 0
time: 43470 steer: 0.9893667698 accel: 1 brake: 0
time: 43480 steer: 1 accel: 1 brake: 0
time: 43570 steer: 0.9874941111 accel: 1 brake: 0
time: 43580 steer: 0.9971123338 accel: 1 brake: 0
time: 43590 steer: 0.1800000072 accel: 1 brake: 0
time: 43740 steer: 0.1838572472 accel: 1 brake: 0
time: 43750 steer: 1 accel: 1 brake: 0
time: 43840 steer: 0.9902420044 accel: 1 brake: 0
time: 43850 steer: 1 accel: 1 brake: 0
time: 43980 steer: 0.9954362512 accel: 1 brake: 0
time: 43990 steer: 1 accel: 1 brake: 0
time: 44150 steer: 1 accel: 1 brake: 1
time: 44430 steer: 0.9858852029 accel: 1 brake: 1
time: 44440 steer: 1 accel: 1 brake: 1
time: 44570 steer: 0.9943153262 accel: 1 brake: 1
time: 44580 steer: 1 accel: 1 brake: 1
time: 44600 steer: 1 accel: 1 brake: 0
time: 45200 steer: 1 accel: 1 brake: 1
time: 45430 steer: 0.9972512126 accel: 1 brake: 1
time: 45440 steer: 1 accel: 1 brake: 1
time: 45450 steer: 1 accel: 1 brake: 0
time: 45530 steer: 0.9948884845 accel: 1 brake: 0
time: 45540 steer: 1 accel: 1 brake: 0
time: 45780 steer: 0.995498836 accel: 1 brake: 0
time: 45790 steer: 1 accel: 1 brake: 0
time: 45930 steer: 0.9921194911 accel: 1 brake: 0
time: 45940 steer: 1 accel: 1 brake: 0
time: 46140 steer: 0.9948256016 accel: 1 brake: 0
time: 46150 steer: 1 accel: 1 brake: 0
time: 46760 steer: 0.9985870123 accel: 1 brake: 0
time: 46770 steer: 1 accel: 1 brake: 0
time: 46880 steer: 0.9996243119 accel: 1 brake: 0
time: 46890 steer: 1 accel: 1 brake: 0
time: 46910 steer: 0.9981209636 accel: 1 brake: 0
time: 46920 steer: 1 accel: 1 brake: 0
time: 47150 steer: 0.9992880225 accel: 1 brake: 0
time: 47160 steer: 1 accel: 1 brake: 0
time: 47320 steer: 0.9957545996 accel: 1 brake: 0
time: 47330 steer: 1 accel: 1 brake: 0
time: 47380 steer: 0.9928547144 accel: 1 brake: 0
time: 47390 steer: 1 accel: 1 brake: 0
time: 47420 steer: 0.998149991 accel: 1 brake: 0
time: 47430 steer: 1 accel: 1 brake: 0
time: 47650 steer: 0.9989071488 accel: 1 brake: 0
time: 47660 steer: 1 accel: 1 brake: 0
time: 47670 steer: 0.9910968542 accel: 1 brake: 0
time: 47680 steer: 1 accel: 1 brake: 0
time: 47790 steer: 0.9927982688 accel: 1 brake: 0
time: 47800 steer: 1 accel: 1 brake: 0
time: 47950 steer: 0.9966436028 accel: 1 brake: 0
time: 47960 steer: 1 accel: 1 brake: 0
time: 48070 steer: 0.9946815372 accel: 1 brake: 0
time: 48080 steer: 1 accel: 1 brake: 0
time: 48100 steer: 0.9844666719 accel: 1 brake: 0
time: 48110 steer: 1 accel: 1 brake: 0
time: 48130 steer: 0.9925226569 accel: 1 brake: 0
time: 48140 steer: 1 accel: 1 brake: 0
time: 48220 steer: 0.9942213297 accel: 1 brake: 0
time: 48230 steer: 1 accel: 1 brake: 0
time: 48270 steer: 0.9937269092 accel: 1 brake: 0
time: 48280 steer: 1 accel: 1 brake: 0
time: 48290 steer: 0.9999435544 accel: 1 brake: 0
time: 48300 steer: 1 accel: 1 brake: 0
time: 48390 steer: 0.8000000119 accel: 1 brake: 0
time: 48400 steer: 0.7950160503 accel: 1 brake: 0
time: 48410 steer: 0.7957161069 accel: 1 brake: 0
time: 48420 steer: 0.7907367349 accel: 1 brake: 0
time: 48430 steer: 0.8000000119 accel: 1 brake: 0
time: 48440 steer: 0.7907959223 accel: 1 brake: 0
time: 48450 steer: 0.8000000119 accel: 1 brake: 0
time: 48480 steer: 0.8072072268 accel: 1 brake: 0
time: 48490 steer: 0.1000000015 accel: 1 brake: 0
time: 48580 steer: 0.09275154769 accel: 1 brake: 0
time: 48590 steer: 0.1092437506 accel: 1 brake: 0
time: 48600 steer: 0.09720908105 accel: 1 brake: 0
time: 48610 steer: 0.1000000015 accel: 1 brake: 0
time: 48620 steer: 0.09770043194 accel: 1 brake: 0
time: 48630 steer: 0.09550401568 accel: 1 brake: 0
time: 48640 steer: 0.1000000015 accel: 1 brake: 0
time: 48700 steer: 0.1000997946 accel: 1 brake: 0
time: 48710 steer: 0.1075060293 accel: 1 brake: 0
time: 48720 steer: 0.1000000015 accel: 1 brake: 0
time: 48800 steer: 0.09808984399 accel: 1 brake: 0
time: 48810 steer: 0.1000000015 accel: 1 brake: 0
time: 48840 steer: 0.09541887045 accel: 1 brake: 0
time: 48850 steer: 0.1000000015 accel: 1 brake: 0
time: 48930 steer: 0.09190984815 accel: 1 brake: 0
time: 48940 steer: 0.200000003 accel: 1 brake: 0
time: 48950 steer: 0.1957960725 accel: 1 brake: 0
time: 48960 steer: 0.200000003 accel: 1 brake: 0
time: 49100 steer: 0.2000000179 accel: 1 brake: 0
time: 49170 steer: 0.194317162 accel: 1 brake: 0
time: 49180 steer: 0.2141673267 accel: 1 brake: 0
time: 49190 steer: 0.2000000179 accel: 1 brake: 0
time: 49200 steer: 0 accel: 1 brake: 0
time: 49550 steer: -0.005127414595 accel: 1 brake: 0
time: 49560 steer: 0.008611407131 accel: 1 brake: 0
time: 49570 steer: 0 accel: 1 brake: 0
time: 49620 steer: 0.004611041397 accel: 1 brake: 0
time: 49630 steer: -0.002231208142 accel: 1 brake: 0
time: 49640 steer: -0.001413006801 accel: 1 brake: 0
time: 49650 steer: 0 accel: 1 brake: 0
time: 49660 steer: 0.002939847298 accel: 1 brake: 0
time: 49670 steer: 0.002344735898 accel: 1 brake: 0
time: 49680 steer: -0.004806359764 accel: 1 brake: 0
time: 49690 steer: 0 accel: 1 brake: 0
time: 49700 steer: 0.004017761908 accel: 1 brake: 0
time: 49710 steer: 0 accel: 1 brake: 0
time: 49750 steer: -3.997888416e-05 accel: 1 brake: 0
time: 49760 steer: 0 accel: 1 brake: 0
time: 49770 steer: -0.00557420589 accel: 1 brake: 0
time: 49780 steer: -0.005889462307 accel: 1 brake: 0
time: 49790 steer: 0 accel: 1 brake: 0
time: 49800 steer: 0.009946897626 accel: 1 brake: 0
time: 49810 steer: 0 accel: 1 brake: 0
time: 49840 steer: -0.008242134005 accel: 1 brake: 0
time: 49850 steer: 0 accel: 1 brake: 0
time: 49900 steer: -0.001034882851 accel: 1 brake: 0
time: 49910 steer: 0 accel: 1 brake: 0
time: 49960 steer: -0.002226325218 accel: 1 brake: 0
time: 49970 steer: 0 accel: 1 brake: 0
time: 50050 steer: 0.007859431207 accel: 1 brake: 0
time: 50060 steer: 0 accel: 1 brake: 0
time: 50100 steer: 0.005091098137 accel: 1 brake: 0
time: 50110 steer: 0 accel: 1 brake: 0
time: 50180 steer: 0.0089574866 accel: 1 brake: 0
time: 50190 steer: 0 accel: 1 brake: 0
time: 50220 steer: -0.009453718551 accel: 1 brake: 0
time: 50230 steer: 0 accel: 1 brake: 0
time: 50270 steer: -0.0009237946942 accel: 1 brake: 0
time: 50280 steer: 0 accel: 1 brake: 0
time: 50370 steer: 0.004246651195 accel: 1 brake: 0
time: 50380 steer: -0.0003622546792 accel: 1 brake: 0
time: 50390 steer: 0 accel: 1 brake: 0
time: 50410 steer: -0.004255195614 accel: 1 brake: 0
time: 50420 steer: 0 accel: 1 brake: 0
time: 50470 steer: 0.003570970148 accel: 1 brake: 0
time: 50480 steer: 0 accel: 1 brake: 0
time: 50520 steer: -0.006021607202 accel: 1 brake: 0
time: 50530 steer: 0 accel: 1 brake: 0
time: 50570 steer: -0.006307870615 accel: 1 brake: 0
time: 50580 steer: 0 accel: 1 brake: 0
time: 50640 steer: 0.0001327553764 accel: 1 brake: 0
time: 50650 steer: 0 accel: 1 brake: 0
time: 50660 steer: 0.00332560204 accel: 1 brake: 0
time: 50670 steer: 0 accel: 1 brake: 0
time: 50690 steer: 0.002079226077 accel: 1 brake: 0
time: 50700 steer: 0 accel: 1 brake: 0
time: 50770 steer: -0.005307473708 accel: 1 brake: 0
time: 50780 steer: 0 accel: 1 brake: 0
time: 50790 steer: -0.006089358125 accel: 1 brake: 0
time: 50800 steer: 0 accel: 1 brake: 0
time: 50830 steer: -0.006565446965 accel: 1 brake: 0
time: 50840 steer: 0 accel: 1 brake: 0
time: 50850 steer: -0.00184667483 accel: 1 brake: 0
time: 50860 steer: 0 accel: 1 brake: 0
time: 50910 steer: 0.003086946905 accel: 1 brake: 0
time: 50920 steer: 0.008557084948 accel: 1 brake: 0
time: 50930 steer: 0 accel: 1 brake: 0
time: 50950 steer: -0.0002890108153 accel: 1 brake: 0
time: 50960 steer: -0.004427320324 accel: 1 brake: 0
time: 50970 steer: 0 accel: 1 brake: 0
time: 50990 steer: 0.00976927951 accel: 1 brake: 0
time: 51000 steer: 0 accel: 1 brake: 0
time: 51060 steer: -0.008201238699 accel: 1 brake: 0
time: 51070 steer: -0.01217017043 accel: 1 brake: 0
time: 51080 steer: 0 accel: 1 brake: 0
time: 51140 steer: 0 accel: 0 brake: 0
"""

# Parse into list of dicts
def parse_input(text):
    lines = text.strip().splitlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        entry = {
            'time': int(parts[1]),
            'steer': float(parts[3]),
            'accel': float(parts[5]),
            'brake': float(parts[7])
        }
        data.append(entry)
    return data

raw_data = parse_input(input_text)

duration_ms = raw_data[-1]['time'] - raw_data[0]['time']

# === INTERPOLATE VALUES PER FRAME ===
total_frames = int((duration_ms / 1000) * fps)
frame_times = np.linspace(0, duration_ms, total_frames)

def interpolate_field(field, t):
    times = [d['time'] for d in raw_data]
    values = [d[field] for d in raw_data]
    return np.interp(t, times, values)

# === DRAW FUNCTIONS ===
def draw_triangle(img, pts, color, white_border=True):
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)
    
    if white_border:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=WHITE, thickness=2)

def draw_box(img, center, size, color, arrow='up'):
    w, h = size
    x, y = center
    top_left = (x - w // 2, y - h // 2)
    bottom_right = (x + w // 2, y + h // 2)
    cv2.rectangle(img, top_left, bottom_right, color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, top_left, bottom_right, WHITE, 2, lineType=cv2.LINE_AA)

    arrow_color = WHITE
    if arrow == 'up':
        pts = [(x, y - 5), (x - 10, y + 5), (x + 10, y + 5)]
    else:
        pts = [(x, y + 5), (x - 10, y - 5), (x + 10, y - 5)]
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], arrow_color)

def draw_visualizer(steer, accel, brake):
    img = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    cx, cy = width - 160, 100

    # Triangles: ← →
    triangle_offset = 35
    triangle_width = 80
    triangle_height = 70
    
    triangle_width_by_steer = abs(int(steer * triangle_width))
    
    draw_triangle(img, [(cx - triangle_offset - triangle_width, cy),
                        (cx - triangle_offset, cy - triangle_height),
                        (cx - triangle_offset, cy + triangle_height)], DARK)
    
    if steer < 0:
        draw_triangle(img, [(cx - triangle_offset - triangle_width_by_steer, cy),
                            (cx - triangle_offset, cy - triangle_height),
                            (cx - triangle_offset, cy + triangle_height)], PINK, False)

    draw_triangle(img, [(cx + triangle_offset + triangle_width, cy),
                        (cx + triangle_offset, cy - triangle_height),
                        (cx + triangle_offset, cy + triangle_height)], DARK)
    if steer > 0:
        draw_triangle(img, [(cx + triangle_offset + triangle_width_by_steer, cy),
                            (cx + triangle_offset, cy - triangle_height),
                            (cx + triangle_offset, cy + triangle_height)], PINK, False)

    # Boxes ↑ ↓
    draw_box(img, (cx, cy - 50), (40, 60), PINK if accel > 0.1 else DARK, 'up')
    draw_box(img, (cx, cy + 50), (40, 60), PINK if brake > 0.1 else DARK, 'down')

    return img


def get_held_value(field, t):
    for i in range(len(raw_data) - 1, -1, -1):
        if t >= raw_data[i]['time']:
            return raw_data[i][field]
    return raw_data[0][field]  # Fallback to first value

# === WRITE VIDEO ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

for t in frame_times:
    steer = get_held_value('steer', t)
    accel = get_held_value('accel', t)
    brake = get_held_value('brake', t)
    frame = draw_visualizer(steer, accel, brake)
    out.write(frame)

out.release()
print("✔ Video saved as:", output_file)