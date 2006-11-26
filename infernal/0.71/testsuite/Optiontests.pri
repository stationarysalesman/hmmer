# This file incorporates Blocks9.plib, the UCSC mixture
# Dirichlet prior created by Kimmen Sjolander.
#

Dirichlet   	# Strategy (mixture Dirichlet)
Amino 		# type of prior (Amino or Nucleic)

# Transitions
1                     # Single component
1.0                   #   with probability = 1.0
0.7939 0.0278 0.0135  # m->m, m->i, m->d alpha's
0.1551 0.1331         # i->m, i->i alpha's 
0.9002 0.5630         # d->m, d->d alpha's

# Match emissions
#
9	# 9 components

0.178091
0.270671 0.039848 0.017576 0.016415 0.014268 0.131916 0.012391 0.022599 0.020358 0.030727 0.015315 0.048298 0.053803 0.020662 0.023612 0.216147 0.147226 0.065438 0.003758 0.009621
# S A T , C G P >< N V M , Q H R I K F L D W , E Y

0.056591
0.021465 0.0103 0.011741 0.010883 0.385651 0.016416 0.076196 0.035329 0.013921 0.093517 0.022034 0.028593 0.013086 0.023011 0.018866 0.029156 0.018153 0.0361 0.07177 0.419641
# Y , F W , H ,>< L M , N Q I C V S R , T P A K D G E

0.0960191
0.561459 0.045448 0.438366 0.764167 0.087364 0.259114 0.21494 0.145928 0.762204 0.24732 0.118662 0.441564 0.174822 0.53084 0.465529 0.583402 0.445586 0.22705 0.02951 0.12109
# Q E , K N R S H D T A >< M P Y G , V L I W C F

0.0781233
0.070143 0.01114 0.019479 0.094657 0.013162 0.048038 0.077 0.032939 0.576639 0.072293 0.02824 0.080372 0.037661 0.185037 0.506783 0.073732 0.071587 0.042532 0.011254 0.028723
# K R , Q , H >< N E T M S , P W Y A L G V C I , D F

0.0834977
0.041103 0.014794 0.00561 0.010216 0.153602 0.007797 0.007175 0.299635 0.010849 0.999446 0.210189 0.006127 0.013021 0.019798 0.014509 0.012049 0.035799 0.180085 0.012744 0.026466
# L M , I , F V ><, W Y C T Q , A P H R , K S E N , D G

0.0904123
0.115607 0.037381 0.012414 0.018179 0.051778 0.017255 0.004911 0.796882 0.017074 0.285858 0.075811 0.014548 0.015092 0.011382 0.012696 0.027535 0.088333 0.94434 0.004373 0.016741
# I V ,, L M >< C T A , F , Y S P W N , E Q K R D G H

0.114468
0.093461 0.004737 0.387252 0.347841 0.010822 0.105877 0.049776 0.014963 0.094276 0.027761 0.01004 0.187869 0.050018 0.110039 0.038668 0.119471 0.065802 0.02543 0.003215 0.018742
# D , E N , Q H S >< K G P T A , R Y , M V L F W I C

0.0682132
0.452171 0.114613 0.06246 0.115702 0.284246 0.140204 0.100358 0.55023 0.143995 0.700649 0.27658 0.118569 0.09747 0.126673 0.143634 0.278983 0.358482 0.66175 0.061533 0.199373
# M , V I L F T Y C A >< W S H Q R N K , P E G , D

0.234585
0.005193 0.004039 0.006722 0.006121 0.003468 0.016931 0.003647 0.002184 0.005019 0.00599 0.001473 0.004158 0.009055 0.00363 0.006583 0.003172 0.00369 0.002967 0.002772 0.002686
# P G W , C H R D E >< N Q K F Y T L A M , S V I


## Insert emissions
1                   # Single component
1.0                 #    with probability 1.0
681 120 623 651 313 902 241 371 687 676 143 548 647 415 551 926 623 505 102 269
