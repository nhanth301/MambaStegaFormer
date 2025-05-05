import pandas as pd

data = {
    "L2": [4.4331, 0.0368, 0.0187, 0.0193],
    "SSIM": [0.2033, 0.3818, 0.4796, 0.5945],
    "LPIPS": [0.3684, 0.4614, 0.3323, 0.3802]
}
df = pd.DataFrame(data, index=["Gatys", "AdaIN", "Two-Stage", "End-to-End"])
print(df)
