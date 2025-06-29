\ Model ilog.cplex
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - t
Subject To
 c1: 4 x2 + 2 x3 + 3 x4 + 5 x5 + x6 >= 5
 c2: 4 x2 + 2 x3 + 3 x4 + 5 x5 + x6 <= 12
 c3: 3 x2 + x5 - 2 x7 - 2 x8 - 2 x9 <= 0
 c4: x3 + x6 - x7 - x8 - x9 - 2 x10 - x11 <= 0
 c5: x2 + x4 + 2 x5 - x7 - 2 x8 - 2 x11 <= 0
 c6: x3 + x5 - x9 - x10 - x11 <= 0
 c7: 2 x4 + x5 - x7 - x8 - x9 - 2 x10 - 2 x11 <= 0
 c8: x7 + x8 + x9 + x10 + x11 - Avar = 0
 c9: Avar - A_hat_0 - A_hat_1 - A_hat_2 - A_hat_3 - A_hat_4 = 0
 c10: t - t_hat_0 - t_hat_1 - t_hat_2 - t_hat_3 - t_hat_4 = 0
 c11: - 4 x2 - 2 x3 - 3 x4 - 5 x5 - x6 + w <= 0
 c12: z_0 + z_1 + z_2 + z_3 + z_4 = 1
 c13: x2 + x3 + x4 + x5 + x6 <= 4
 c14: x2 + x3 + x4 + x5 + x6 >= 1
 c15: t >= -1e-05
 c16: t_hat_0 - 2.5 z_0 >= 0
 c17: t_hat_0 - 5.94117647058824 z_0 <= 0
 c18: A_hat_0 - z_0 >= 0
 c19: A_hat_0 - 2 z_0 <= 0
 c20: t_hat_1 - 1.66666666666667 z_1 >= 0
 c21: t_hat_1 - 6 z_1 <= 0
 c22: A_hat_1 - 2 z_1 >= 0
 c23: A_hat_1 - 3 z_1 <= 0
 c24: t_hat_2 - 1.25 z_2 >= 0
 c25: t_hat_2 - 4 z_2 <= 0
 c26: A_hat_2 - 3 z_2 >= 0
 c27: A_hat_2 - 4 z_2 <= 0
 c28: t_hat_3 - z_3 >= 0
 c29: t_hat_3 - 3 z_3 <= 0
 c30: A_hat_3 - 4 z_3 >= 0
 c31: A_hat_3 - 5 z_3 <= 0
 c32: t_hat_4 - 0.833333333333333 z_4 >= 0
 c33: t_hat_4 - 2.4 z_4 <= 0
 c34: A_hat_4 - 5 z_4 >= 0
 c35: A_hat_4 - 6 z_4 <= 0
 c36: - 5.94117647058824 A_hat_0 - 6 A_hat_1 - 4 A_hat_2 - 3 A_hat_3
   - 2.4 A_hat_4 - 2 t_hat_0 - 3 t_hat_1 - 4 t_hat_2 - 5 t_hat_3
   - 6 t_hat_4 + w + 11.8823529411765 z_0 + 18 z_1 + 16 z_2 + 15 z_3
   + 14.4 z_4 >= 0
 c37: - 2 A_hat_0 - A_hat_1 - A_hat_2 - A_hat_3 - t_hat_0 - 2 t_hat_1
   - 3 t_hat_2 - 4 t_hat_3 - 5 t_hat_4 + w + 2 z_0 + 2 z_1 + 3 z_2 + 4 z_3
   >= 0
 c38: - 2 A_hat_0 - A_hat_1 - A_hat_2 - A_hat_3 - 2 t_hat_0 - 3 t_hat_1
   - 4 t_hat_2 - 5 t_hat_3 - 6 t_hat_4 + w + 4 z_0 + 3 z_1 + 4 z_2 + 5 z_3
   <= 0
 c39: - 5.94117647058824 A_hat_0 - 6 A_hat_1 - 4 A_hat_2 - 3 A_hat_3
   - 2.4 A_hat_4 - t_hat_0 - 2 t_hat_1 - 3 t_hat_2 - 4 t_hat_3 - 5 t_hat_4
   + w + 5.94117647058824 z_0 + 12 z_1 + 12 z_2 + 12 z_3 + 12 z_4 <= 0
Bounds
 1 <= t <= 12
 Avar <= 5
 A_hat_0 <= 5
 A_hat_1 <= 5
 A_hat_2 <= 5
 A_hat_3 <= 5
 A_hat_4 <= 5
 t_hat_0 <= 12
 t_hat_1 <= 12
 t_hat_2 <= 12
 t_hat_3 <= 12
 t_hat_4 <= 12
 w <= 60
Binaries
 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 z_0 z_1 z_2 z_3 z_4
End
