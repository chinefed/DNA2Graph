import signal
import multiprocessing as mp

from dna2graph.utils import check_for_updates
from dna2graph.gui.main_window import MainWindow


def main():
    mp.freeze_support()
    
    # Check for updates
    check_for_updates()

    # Main window
    root = MainWindow()

    def sigint_handler(signum, frame):
        '''
        Handles shutdown in case of CTRL-C event.
        '''
        root.stop_event.set()
        root.destroy()

    signal.signal(signal.SIGINT, sigint_handler)

    root.mainloop()

if __name__ == '__main__':
    main()

