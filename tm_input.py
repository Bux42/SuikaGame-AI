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
BG_COLOR = (10, 10, 20)

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
time: 26900 steer: 0.496062994 accel: 1 brake: 0
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
time: 27030 steer: 0.006815697998 accel: 1 brake: 0
time: 27040 steer: -0.0463225171 accel: 1 brake: 0
time: 27050 steer: 0 accel: 1 brake: 0
time: 27070 steer: -0.002878810745 accel: 1 brake: 0
time: 27080 steer: 0 accel: 1 brake: 0
time: 27100 steer: 0.008742026985 accel: 1 brake: 0
time: 27110 steer: 0 accel: 1 brake: 0
time: 27140 steer: 0.03833276778 accel: 1 brake: 0
time: 27150 steer: 0 accel: 1 brake: 0
time: 27170 steer: 0.007037872449 accel: 1 brake: 0
time: 27180 steer: 0.00166173093 accel: 1 brake: 0
time: 27190 steer: 0 accel: 1 brake: 0
time: 27200 steer: 0.00839747116 accel: 1 brake: 0
time: 27210 steer: 0 accel: 1 brake: 0
time: 27220 steer: -0.01044495776 accel: 1 brake: 0
time: 27230 steer: -0.001128878444 accel: 1 brake: 0
time: 27240 steer: -0.01578691974 accel: 1 brake: 0
time: 27250 steer: 0.05751091242 accel: 1 brake: 0
time: 27260 steer: 0 accel: 1 brake: 0
time: 27280 steer: 0.009764401242 accel: 1 brake: 0
time: 27290 steer: 0 accel: 1 brake: 0
time: 27320 steer: 0.06299212575 accel: 1 brake: 0
time: 27330 steer: 0.1417322904 accel: 1 brake: 0
time: 27340 steer: 0.2204724401 accel: 1 brake: 0
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
time: 27570 steer: 0.6111345291 accel: 1 brake: 0
time: 27580 steer: 0.6151208282 accel: 1 brake: 0
time: 27590 steer: 0.6274276972 accel: 1 brake: 0
time: 27600 steer: 0.8144465685 accel: 1 brake: 0
time: 27610 steer: 0.8602145314 accel: 1 brake: 0
time: 27620 steer: 0.814314723 accel: 1 brake: 0
time: 27630 steer: 0.7818042636 accel: 1 brake: 0
time: 27640 steer: 0.8236442208 accel: 1 brake: 0
time: 27650 steer: 0.827665329 accel: 1 brake: 0
time: 27660 steer: 0.8000000119 accel: 1 brake: 0
time: 27670 steer: 0.8283721805 accel: 1 brake: 0
time: 27680 steer: 0.8444501758 accel: 1 brake: 0
time: 27690 steer: 0.8062990308 accel: 1 brake: 0
time: 27700 steer: 0.832542479 accel: 1 brake: 0
time: 27710 steer: 0.7861119509 accel: 1 brake: 0
time: 27720 steer: 0.7664009929 accel: 1 brake: 0
time: 27730 steer: 0.8061046004 accel: 1 brake: 0
time: 27740 steer: 0.7790759206 accel: 1 brake: 0
time: 27750 steer: 0.8127964735 accel: 1 brake: 0
time: 27760 steer: 0.8048607111 accel: 1 brake: 0
time: 27770 steer: 0.8000000119 accel: 1 brake: 0
time: 27790 steer: 0.7424695492 accel: 1 brake: 0
time: 27800 steer: 0.788887918 accel: 1 brake: 0
time: 27810 steer: 0.7943034172 accel: 1 brake: 0
time: 27820 steer: 0.8240504265 accel: 1 brake: 0
time: 27830 steer: 0.8000000119 accel: 1 brake: 0
time: 27840 steer: 0.8165871501 accel: 1 brake: 0
time: 27850 steer: 0.794455409 accel: 1 brake: 0
time: 27860 steer: 0.8419227004 accel: 1 brake: 0
time: 27870 steer: 0.7975771427 accel: 1 brake: 0
time: 27880 steer: 0.828884244 accel: 1 brake: 0
time: 27890 steer: 0.8000000119 accel: 1 brake: 0
time: 27900 steer: 0.7063698173 accel: 1 brake: 0
time: 27910 steer: 0.7571556568 accel: 1 brake: 0
time: 27920 steer: 0.7968111038 accel: 1 brake: 0
time: 27930 steer: 0.8000000119 accel: 1 brake: 0
time: 27940 steer: 0.796756804 accel: 1 brake: 0
time: 27950 steer: 0.796163559 accel: 1 brake: 0
time: 27960 steer: 0.8733521104 accel: 1 brake: 0
time: 27970 steer: 0.7977358699 accel: 1 brake: 0
time: 27980 steer: 0.8025256991 accel: 1 brake: 0
time: 27990 steer: 0.8000000119 accel: 1 brake: 0
time: 28010 steer: 0.8037641644 accel: 1 brake: 0
time: 28020 steer: 0.8390625119 accel: 1 brake: 0
time: 28030 steer: 0.7745091319 accel: 1 brake: 0
time: 28040 steer: 0.7820175886 accel: 1 brake: 0
time: 28050 steer: 0.8303369284 accel: 1 brake: 0
time: 28060 steer: 0.8000000119 accel: 1 brake: 0
time: 28070 steer: 0.787379086 accel: 1 brake: 0
time: 28080 steer: 0.8255211115 accel: 1 brake: 0
time: 28090 steer: 0.7734641433 accel: 1 brake: 0
time: 28100 steer: 0.7967089415 accel: 1 brake: 0
time: 28110 steer: 0.8401678801 accel: 1 brake: 0
time: 28120 steer: 0.7621690035 accel: 1 brake: 0
time: 28130 steer: 0.8000000119 accel: 1 brake: 0
time: 28140 steer: 0.8134871125 accel: 1 brake: 0
time: 28150 steer: 0.8000000119 accel: 1 brake: 0
time: 28160 steer: 0.8217939138 accel: 1 brake: 0
time: 28170 steer: 0.7888028026 accel: 1 brake: 0
time: 28180 steer: 0.7955147028 accel: 1 brake: 0
time: 28190 steer: 0.7968422771 accel: 1 brake: 0
time: 28200 steer: 0.7463820577 accel: 1 brake: 0
time: 28210 steer: 0.7690343857 accel: 1 brake: 0
time: 28220 steer: 0.8070561886 accel: 1 brake: 0
time: 28230 steer: 0.8291351199 accel: 1 brake: 0
time: 28240 steer: 0.784101367 accel: 1 brake: 0
time: 28250 steer: 0.7996536493 accel: 1 brake: 0
time: 28260 steer: 0.8257933259 accel: 1 brake: 0
time: 28270 steer: 0.8285989165 accel: 1 brake: 0
time: 28280 steer: 0.82858181 accel: 1 brake: 0
time: 28290 steer: 0.8000000119 accel: 1 brake: 0
time: 28300 steer: 0.8144156933 accel: 1 brake: 0
time: 28310 steer: 0.7948426604 accel: 1 brake: 0
time: 28320 steer: 0.7929065824 accel: 1 brake: 0
time: 28330 steer: 0.8000000119 accel: 1 brake: 0
time: 28340 steer: 0.8249205351 accel: 1 brake: 0
time: 28350 steer: 0.8466979265 accel: 1 brake: 0
time: 28360 steer: 0.7955855131 accel: 1 brake: 0
time: 28370 steer: 0.809663713 accel: 1 brake: 0
time: 28380 steer: 0.7835044265 accel: 1 brake: 0
time: 28390 steer: 0.8273217082 accel: 1 brake: 0
time: 28400 steer: 0.814186573 accel: 1 brake: 0
time: 28410 steer: 0.8000000119 accel: 1 brake: 0
time: 28420 steer: 0.8346431255 accel: 1 brake: 0
time: 28430 steer: 0 accel: 1 brake: 0
time: 28480 steer: -0.02921628393 accel: 1 brake: 0
time: 28490 steer: 0 accel: 1 brake: 0
time: 28500 steer: -0.03523239866 accel: 1 brake: 0
time: 28510 steer: 0 accel: 1 brake: 0
time: 28540 steer: -0.02093050815 accel: 1 brake: 0
time: 28550 steer: 0.002432631329 accel: 1 brake: 0
time: 28560 steer: 0 accel: 1 brake: 0
time: 28570 steer: -1 accel: 1 brake: 0
time: 28610 steer: -0.9827610254 accel: 1 brake: 0
time: 28620 steer: -1 accel: 1 brake: 0
time: 28670 steer: -0.9861064553 accel: 1 brake: 0
time: 28680 steer: -1 accel: 1 brake: 0
time: 28920 steer: -0.9912701845 accel: 1 brake: 0
time: 28930 steer: -1 accel: 1 brake: 0
time: 29210 steer: -0.9886168838 accel: 1 brake: 0
time: 29220 steer: -1 accel: 1 brake: 0
time: 29630 steer: -0.9852348566 accel: 1 brake: 0
time: 29640 steer: -1 accel: 1 brake: 0
time: 29870 steer: -0.9992922544 accel: 1 brake: 0
time: 29880 steer: -1 accel: 1 brake: 0
time: 30200 steer: -1 accel: 1 brake: 1
time: 30310 steer: -0.9928961992 accel: 1 brake: 1
time: 30320 steer: -1 accel: 1 brake: 1
time: 30660 steer: -0.9897595048 accel: 1 brake: 1
time: 30670 steer: -1 accel: 1 brake: 1
time: 30710 steer: -0.971226871 accel: 1 brake: 1
time: 30720 steer: -0.9708441496 accel: 1 brake: 1
time: 30730 steer: -1 accel: 1 brake: 1
time: 30750 steer: -1 accel: 1 brake: 0
time: 30780 steer: -0.9860130548 accel: 1 brake: 0
time: 30790 steer: -1 accel: 1 brake: 0
time: 31060 steer: -0.9989022613 accel: 1 brake: 0
time: 31070 steer: -1 accel: 1 brake: 0
time: 31220 steer: -0.9961830378 accel: 1 brake: 0
time: 31230 steer: -1 accel: 1 brake: 0
time: 31480 steer: -0.9943208098 accel: 1 brake: 0
time: 31490 steer: -0.9786263704 accel: 1 brake: 0
time: 31500 steer: -1 accel: 1 brake: 0
time: 31540 steer: -0.9993014336 accel: 1 brake: 0
time: 31550 steer: -1 accel: 1 brake: 0
time: 31570 steer: -0.9942347407 accel: 1 brake: 0
time: 31580 steer: -1 accel: 1 brake: 0
time: 31670 steer: -0.9926187992 accel: 1 brake: 0
time: 31680 steer: -1 accel: 1 brake: 0
time: 31700 steer: -0.9996255636 accel: 1 brake: 0
time: 31710 steer: -1 accel: 1 brake: 0
time: 31780 steer: -0.9982668757 accel: 1 brake: 0
time: 31790 steer: -1 accel: 1 brake: 0
time: 31800 steer: -0.9874824882 accel: 1 brake: 0
time: 31810 steer: -0.9819186926 accel: 1 brake: 0
time: 31820 steer: -1 accel: 1 brake: 0
time: 31840 steer: -0.9947547913 accel: 1 brake: 0
time: 31850 steer: -1 accel: 1 brake: 0
time: 31860 steer: -0.9837937355 accel: 1 brake: 0
time: 31870 steer: -1 accel: 1 brake: 0
time: 31960 steer: -0.9993856549 accel: 1 brake: 0
time: 31970 steer: -1 accel: 1 brake: 0
time: 32040 steer: -0.9859874249 accel: 1 brake: 0
time: 32050 steer: -1 accel: 1 brake: 0
time: 32060 steer: -0.9962764382 accel: 1 brake: 0
time: 32070 steer: -1 accel: 1 brake: 0
time: 32080 steer: -0.981611073 accel: 1 brake: 0
time: 32090 steer: -1 accel: 1 brake: 0
time: 32190 steer: -0.9769197702 accel: 1 brake: 0
time: 32200 steer: -1 accel: 1 brake: 0
time: 32210 steer: -0.9887945056 accel: 1 brake: 0
time: 32220 steer: -1 accel: 1 brake: 0
time: 32230 steer: -0.9874321818 accel: 1 brake: 0
time: 32240 steer: -1 accel: 1 brake: 0
time: 32340 steer: -0.972070992 accel: 1 brake: 0
time: 32350 steer: -1 accel: 1 brake: 0
time: 32400 steer: -0.9707965255 accel: 1 brake: 0
time: 32410 steer: -1 accel: 1 brake: 0
time: 32430 steer: -0.9753889441 accel: 1 brake: 0
time: 32440 steer: -0.9784011245 accel: 1 brake: 0
time: 32450 steer: -0.9747627378 accel: 1 brake: 0
time: 32460 steer: -1 accel: 1 brake: 0
time: 32520 steer: -0.9945716858 accel: 1 brake: 0
time: 32530 steer: -1 accel: 1 brake: 0
time: 32540 steer: -0.9733912349 accel: 1 brake: 0
time: 32550 steer: -0.9845884442 accel: 1 brake: 0
time: 32560 steer: -1 accel: 1 brake: 0
time: 32610 steer: -0.9917004704 accel: 1 brake: 0
time: 32620 steer: -0.9876655936 accel: 1 brake: 0
time: 32630 steer: -0.9562804103 accel: 1 brake: 0
time: 32640 steer: -0.9539274573 accel: 1 brake: 0
time: 32650 steer: -0.9823572636 accel: 1 brake: 0
time: 32660 steer: -0.985119462 accel: 1 brake: 0
time: 32670 steer: -0.9787673354 accel: 1 brake: 0
time: 32680 steer: -0.9620007873 accel: 1 brake: 0
time: 32690 steer: -0.9897576571 accel: 1 brake: 0
time: 32700 steer: -1 accel: 1 brake: 0
time: 32710 steer: -0.9626709819 accel: 1 brake: 0
time: 32720 steer: -0.9714850187 accel: 1 brake: 0
time: 32730 steer: -0.9572325945 accel: 1 brake: 0
time: 32740 steer: -0.9792681932 accel: 1 brake: 0
time: 32750 steer: -1 accel: 1 brake: 0
time: 32780 steer: -0.9824497104 accel: 1 brake: 0
time: 32790 steer: -1 accel: 1 brake: 0
time: 32800 steer: -0.9715473056 accel: 1 brake: 0
time: 32810 steer: -0.9419620633 accel: 1 brake: 0
time: 32820 steer: -0.9342191815 accel: 1 brake: 0
time: 32830 steer: -0.9459181428 accel: 1 brake: 0
time: 32840 steer: -0.9713110924 accel: 1 brake: 0
time: 32850 steer: -0.9775927067 accel: 1 brake: 0
time: 32860 steer: -0.9344764948 accel: 1 brake: 0
time: 32870 steer: -0.9788222909 accel: 1 brake: 0
time: 32880 steer: -1 accel: 1 brake: 0
time: 32890 steer: -0.9207752347 accel: 1 brake: 0
time: 32900 steer: -1 accel: 1 brake: 0
time: 32910 steer: -0.9672642946 accel: 1 brake: 0
time: 32920 steer: -0.9659010768 accel: 1 brake: 0
time: 32930 steer: -0.9333127737 accel: 1 brake: 0
time: 32940 steer: -1 accel: 1 brake: 0
time: 32960 steer: -0.9515909553 accel: 1 brake: 0
time: 32970 steer: -0.942027986 accel: 1 brake: 0
time: 32980 steer: -0.9886983633 accel: 1 brake: 0
time: 32990 steer: -0.9962728024 accel: 1 brake: 0
time: 33000 steer: -0.962163806 accel: 1 brake: 0
time: 33010 steer: -0.9643263221 accel: 1 brake: 0
time: 33020 steer: -0.7910284996 accel: 1 brake: 0
time: 33030 steer: -0.7707599401 accel: 1 brake: 0
time: 33040 steer: -0.7397720218 accel: 1 brake: 0
time: 33050 steer: -0.7681963444 accel: 1 brake: 0
time: 33060 steer: -0.7724280953 accel: 1 brake: 0
time: 33070 steer: -0.8000000119 accel: 1 brake: 0
time: 33080 steer: -0.7855864167 accel: 1 brake: 0
time: 33090 steer: -0.7559416294 accel: 1 brake: 0
time: 33100 steer: -0.8000000119 accel: 1 brake: 0
time: 33110 steer: -0.8353532553 accel: 1 brake: 0
time: 33120 steer: -0.8026716113 accel: 1 brake: 0
time: 33130 steer: -0.7619642019 accel: 1 brake: 0
time: 33140 steer: -0.7821842432 accel: 1 brake: 0
time: 33150 steer: -0.7901220918 accel: 1 brake: 0
time: 33160 steer: -0.7851268053 accel: 1 brake: 0
time: 33170 steer: -0.7774499059 accel: 1 brake: 0
time: 33180 steer: -0.7638657689 accel: 1 brake: 0
time: 33190 steer: -0.8067724109 accel: 1 brake: 0
time: 33200 steer: -0.7611969113 accel: 1 brake: 0
time: 33210 steer: -0.7891982794 accel: 1 brake: 0
time: 33220 steer: -0.7574065924 accel: 1 brake: 0
time: 33230 steer: -0.8000000119 accel: 1 brake: 0
time: 33240 steer: -0.8040312529 accel: 1 brake: 0
time: 33250 steer: -0.8334361315 accel: 1 brake: 0
time: 33260 steer: -0.7901568413 accel: 1 brake: 0
time: 33270 steer: -0.7321198583 accel: 1 brake: 0
time: 33280 steer: -0.7843834162 accel: 1 brake: 0
time: 33290 steer: -0.8000000119 accel: 1 brake: 0
time: 33310 steer: -0.7516312003 accel: 1 brake: 0
time: 33320 steer: -0.7824469805 accel: 1 brake: 0
time: 33330 steer: -0.8002389669 accel: 1 brake: 0
time: 33340 steer: -0.7806744576 accel: 1 brake: 0
time: 33350 steer: -0.7840739489 accel: 1 brake: 0
time: 33360 steer: -0.8000000119 accel: 1 brake: 0
time: 33380 steer: -0.8282448649 accel: 1 brake: 0
time: 33390 steer: -0.8775658607 accel: 1 brake: 0
time: 33400 steer: -0.8277129531 accel: 1 brake: 0
time: 33410 steer: -0.8495489359 accel: 1 brake: 0
time: 33420 steer: -0.8002169728 accel: 1 brake: 0
time: 33430 steer: -0.7608252168 accel: 1 brake: 0
time: 33440 steer: -0.8008221388 accel: 1 brake: 0
time: 33450 steer: -0.8000000119 accel: 1 brake: 0
time: 33460 steer: -0.8149748445 accel: 1 brake: 0
time: 33470 steer: -0.872380197 accel: 1 brake: 0
time: 33480 steer: -0.7439525127 accel: 1 brake: 0
time: 33490 steer: -0.7694817781 accel: 1 brake: 0
time: 33500 steer: -0.7827518582 accel: 1 brake: 0
time: 33510 steer: -0.770047605 accel: 1 brake: 0
time: 33520 steer: -0.8443595767 accel: 1 brake: 0
time: 33530 steer: -0.7594271898 accel: 1 brake: 0
time: 33540 steer: -0.8081054091 accel: 1 brake: 0
time: 33550 steer: -0.8449107409 accel: 1 brake: 0
time: 33560 steer: -0.8016452789 accel: 1 brake: 0
time: 33570 steer: -0.6984630823 accel: 1 brake: 0
time: 33580 steer: -0.7725406885 accel: 1 brake: 0
time: 33590 steer: -0.7893044949 accel: 1 brake: 0
time: 33600 steer: 0.05426862463 accel: 1 brake: 0
time: 33610 steer: -0.01100588962 accel: 1 brake: 0
time: 33620 steer: -0.01040528528 accel: 1 brake: 0
time: 33630 steer: -0.02209143341 accel: 1 brake: 0
time: 33640 steer: -0.002557147294 accel: 1 brake: 0
time: 33650 steer: -0.03903561085 accel: 1 brake: 0
time: 33660 steer: -0.009550156072 accel: 1 brake: 0
time: 33670 steer: -0.02763603628 accel: 1 brake: 0
time: 33680 steer: 0.03672108799 accel: 1 brake: 0
time: 33690 steer: 0.01712729223 accel: 1 brake: 0
time: 33700 steer: 0.008162174374 accel: 1 brake: 0
time: 33710 steer: 0.01134464704 accel: 1 brake: 0
time: 33720 steer: -0.08146427572 accel: 1 brake: 0
time: 33730 steer: -0.0159764383 accel: 1 brake: 0
time: 33740 steer: -0.004718773067 accel: 1 brake: 0
time: 33750 steer: -0.003136692569 accel: 1 brake: 0
time: 33760 steer: 0.03367137536 accel: 1 brake: 0
time: 33770 steer: 0.0007022321224 accel: 1 brake: 0
time: 33780 steer: -0.02232216112 accel: 1 brake: 0
time: 33790 steer: 0 accel: 1 brake: 0
time: 33800 steer: 0.02477034926 accel: 1 brake: 0
time: 33810 steer: -0.02683217824 accel: 1 brake: 0
time: 33820 steer: 0.01841090061 accel: 1 brake: 0
time: 33830 steer: -0.03687398881 accel: 1 brake: 0
time: 33840 steer: 0.008521070704 accel: 1 brake: 0
time: 33850 steer: -0.01213202439 accel: 1 brake: 0
time: 33860 steer: 0.01103610452 accel: 1 brake: 0
time: 33870 steer: 0.06849360466 accel: 1 brake: 0
time: 33880 steer: 0.02884090692 accel: 1 brake: 0
time: 33890 steer: -0.001843929291 accel: 1 brake: 0
time: 33900 steer: 0 accel: 0 brake: 0
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