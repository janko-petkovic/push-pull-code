import tkinter as tk
from guisrc.windows import MainWindow
import matplotlib.pyplot as plt
plt.style.use('guisrc.windows.guistyle')


def main():
    # This is needed for quitting in multi-window guis
    def _quit():
        root.quit()
        root.destroy()

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", _quit)
    main_window = MainWindow(root).pack(side='top', fill='both', expand=True)
    root.mainloop()

if __name__ == '__main__':
    main()
