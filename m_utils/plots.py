import os
import matplotlib.pyplot as plt

def plot_and_save_loss(train_losses, val_losses, save_dir):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))  # Guardar la figura como un archivo .png
    plt.close()  # Cerrar la figura actual para liberar memoria

def plot_and_save_mae(train_maes, val_maes, save_dir):
    plt.plot(train_maes, label='Training MAE')
    plt.plot(val_maes, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'mae_plot.png'))  # Guardar la figura como un archivo .png
    plt.close()  # Cerrar la figura actual para liberar memoria
