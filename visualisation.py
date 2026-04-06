import os
import matplotlib.pyplot as plt

class TrainingTracker:
    def __init__(self, name="model", output_dir="results"):
        self.name = name
        self.output_dir = output_dir
        self.train_loss = []
        self.val_loss = []
        self.val_f1 = []

        os.makedirs(output_dir, exist_ok=True)

    def log(self, train_loss, val_loss, val_f1):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_f1.append(val_f1)

    def plot(self):
        plt.figure()
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Val Loss")
        plt.title(f"Loss - {self.name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        plt.savefig(f"{self.output_dir}/{self.name}_loss.png")
        plt.close()

    def plot_f1(self):
        plt.figure()
        plt.plot(self.val_f1, label="Val F1")
        plt.title(f"F1 Score - {self.name}")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.legend()
        plt.grid()

        plt.savefig(f"{self.output_dir}/{self.name}_F1score.png")
        plt.close()
