import numpy as np

# Generate some random bits, encode it to valid codewords and simulate transmission
def encode_and_transmission(G_matrix, SNR, batch_size, noise_io, top_config, rng=0):
    K, N = np.shape(G_matrix)
    if rng == 0:
        x_bits = np.random.randint(0, 2, size=(batch_size, K))
    else:
        x_bits = rng.randint(0, 2, size=(batch_size, K))
    # coding
    u_coded_bits = np.mod(np.matmul(x_bits, G_matrix), 2)  # G_matrix

    # BPSK modulation
    s_mod = u_coded_bits * (-2) + 1
    # plus the noise
    ch_noise_normalize = noise_io.generate_noise(batch_size)

    ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
    ch_noise = ch_noise_normalize * ch_noise_sigma
    y_receive = s_mod + ch_noise

    intf_label = np.zeros((batch_size, 1))

    if top_config.urllc_intf:
        intf_label = np.random.choice([0, 1], size=[batch_size, 1], p=[1 - top_config.intf_prob, top_config.intf_prob])

        intf_bits = np.random.randint(0, 2, size=(batch_size, top_config.intf_len))
        intf_mod = intf_bits * (-2) + 1
        intf_signal = np.zeros((batch_size, N))
        for l in range(batch_size):
            if intf_label[l] == 1:
                intf_signal[l, 0:top_config.intf_len] = intf_mod[l, :]

        y_receive = y_receive + intf_signal

    LLR = y_receive * 2.0 / (ch_noise_sigma * ch_noise_sigma)
    return x_bits, u_coded_bits, s_mod, ch_noise, y_receive, LLR, intf_label

class LDPC:
    def __init__(self, N, K, file_G, file_H):
        self.N = N
        self.K = K
        self.G_matrix, self.H_matrix = self.init_LDPC_G_H(file_G, file_H)

    def init_LDPC_G_H(self, file_G, file_H):
        G_matrix_row_col = np.loadtxt(file_G, dtype=np.int32)
        H_matrix_row_col = np.loadtxt(file_H, dtype=np.int32)
        G_matrix = np.zeros([self.K, self.N], dtype=np.int32)
        H_matrix = np.zeros([self.N-self.K, self.N], dtype=np.int32)
        G_matrix[G_matrix_row_col[:, 0], G_matrix_row_col[:, 1]] = 1
        H_matrix[H_matrix_row_col[:, 0], H_matrix_row_col[:, 1]] = 1
        return G_matrix, H_matrix

    def dec_src_bits(self, bp_output):
        return bp_output[:,0:self.K]

