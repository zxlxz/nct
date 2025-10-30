import numpy as np
import matplotlib.pyplot as plt


def load_bin_f32(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(shape)
    return data


def view_2d(X, Y, Z):
    SOD = 611
    SDD = 1114

    plt.grid(True)
    plt.plot(X[32, :], Y[32, :], ".-")
    plt.plot(Z[:, 592], Y[:, 592], ".-")

    px = X[32, 592] - X[32, 591]
    pu = Z[1, :] - Z[0, :]
    print(f"px={px}, {px * SOD / SDD} ")
    print(f"pz={pu}, {pu * SOD / SDD} ")


def view_3d(X, Y, Z):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax.set_box_aspect([1, 1, 1])


def main(data_dir):
    X = load_bin_f32(data_dir / "detpos_x.bin", (64, 1184))
    Y = load_bin_f32(data_dir / "detpos_y.bin", (64, 1184))
    Z = load_bin_f32(data_dir / "detpos_z.bin", (64, 1184))

    view_2d(X, Y, Z)
    # view_3d(X, Y, Z)
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    main(data_dir)
