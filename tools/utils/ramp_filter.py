#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq,fftshift

def make_omega(freq):
    omega = np.abs(freq) * np.pi
    return omega


def make_coswin(omega, exp=1):
    cos_omega = np.clip(np.cos(omega), 0, 1)
    weight = np.pow(cos_omega, 2 * exp)
    return weight


def run(n=1024):
    freq = fftfreq(n)
    omega = make_omega(freq)
    coswin = make_coswin(omega)


    plt.title("Ramp Filter")
    plt.xlabel("Frequency")
    plt.ylabel("Weight")
    plt.grid()

    plt.plot(fftshift(omega), label="Ramp Filter")
    plt.plot(fftshift(coswin), label="Cosine Window")
    plt.plot(fftshift(omega*coswin), label="Ramp + Cosine Window")
    plt.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    run()
