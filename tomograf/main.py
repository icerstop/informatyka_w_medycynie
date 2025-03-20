from gui import CTScannerApp
import tkinter as tk

def main():
    root = tk.Tk()
    app = CTScannerApp(root)
    root.geometry("1200x800")
    root.mainloop()

if __name__ == "__main__":
    main()