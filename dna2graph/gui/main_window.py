import threading
import multiprocessing as mp
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from tqdm import tqdm

from dna2graph.utils import UserCancelledError
from dna2graph.constants import (
    APP_NAME,
    DEVELOPER_EMAIL,
    TK_STYLE,
    INPUT_EXT
)
from dna2graph.config.parameter_manager import ParameterManager
from dna2graph.gui.advanced_preferences import AdvancedPreferencesWindow
from dna2graph.core.app import DNA2Graph


DEFAULT_SAVING_PREF = True # All saving preferences default to True
DEFAULT_CLEAN_CACHE = False


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.protocol('WM_DELETE_WINDOW', self.on_close)
        self.title(APP_NAME)
        self.geometry('720x570')
        self.resizable(False, False)

        self.style = ttk.Style(self)
        self.style.theme_use(TK_STYLE)
        self.style.configure(
            'Start.TButton',
            font=('TkDefaultFont', 14, 'bold')
        )

        self.bg_color = self.style.lookup('TFrame', 'background')
        self.configure(bg=self.bg_color)

        self.params_mgr = ParameterManager()
        self.advanced_window = None
        self.analysis_thread = None
        self.stop_event = mp.Event()

        self._build_ui()

    def _build_ui(self):
        # === Path Selectors ====
        self.input_file_path_var = self._build_path_selector(
            parent=self,
            label_text='Input Images:',
            command=self.select_files,
            fill='x',
            padx=(20, 20),
            pady=(20, 0)
        )
        self.output_root_dir_var = self._build_path_selector(
            parent=self,
            label_text='Output Directory:',
            command=self.select_directory,
            fill='x',
            padx=(20, 20),
            pady=(10, 0)
        )

        # === Algorithm Preferences ===

        algo_pref = ttk.LabelFrame(self, text='Algorithm Preferences')
        algo_pref.pack(fill='x', padx=(20, 20), pady=(20, 0))

        algo_box_frame = ttk.Frame(algo_pref)
        algo_box_frame.pack(fill='x', padx=10, pady=5)

        self.trained_seg = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            algo_box_frame,
            text='Use Trained Segmentation Pipeline (BETA)',
            variable=self.trained_seg
        ).pack(side='left', padx=(10, 0))

        # Advanced Preferences
        ttk.Button(
            algo_box_frame,
            text='Advanced',
            command=self.open_advanced_preferences
        ).pack(side='right')

        # === Export Preferences ===
        export_pref = ttk.LabelFrame(self, text='Export Preferences')
        export_pref.pack(fill='x', padx=(20, 20), pady=(20, 0))

        # Container for left/right layout
        row_frame = ttk.Frame(export_pref)
        row_frame.pack(fill='x', padx=10, pady=10)

        # --- Data Exports (left) ---
        data_frame = ttk.LabelFrame(row_frame, text='Data Exports')
        data_frame.pack(side='left', fill='both', expand=True, padx=5)

        for text, var_name in [
            ('Graph Representation', 'save_graph'),
            ('Segmentation Mask', 'save_mask'),
            ('Report', 'save_report'),
        ]:
            var = self._build_checkbutton(
                data_frame,
                val=DEFAULT_SAVING_PREF,
                text=text,
                side='top',
                anchor='w',
                padx=10,
                pady=2,
            )
            setattr(self, var_name, var)

        # --- ROI Exports (right) ---
        roi_frame = ttk.LabelFrame(row_frame, text='ROI Exports (ImageJ)')
        roi_frame.pack(side='left', fill='both', expand=True, padx=5)

        for text, var_name in [
            ('Segmentation ROIs', 'save_segmentation_rois'),
            ('Bounding Box ROIs', 'save_bbox_rois'),
            ('Linear Walk Decomposition ROIs', 'save_lin_decomp_rois'),
        ]:
            var = self._build_checkbutton(
                roi_frame,
                val=DEFAULT_SAVING_PREF,
                text=text,
                side='top',
                anchor='w',
                padx=10,
                pady=2,
            )
            setattr(self, var_name, var)

        # === Cache Preferences ===

        cache_pref = ttk.LabelFrame(self, text='Cache Preferences')
        cache_pref.pack(fill='x', padx=(20, 20), pady=(20, 0))

        self.clean_cache = self._build_checkbutton(
            cache_pref,
            val=DEFAULT_CLEAN_CACHE,
            text='Clean Cache',
            pady=(5, 5)
        )

        # === Start Analysis ===

        ttk.Button(
            self,
            text='Start Analysis',
            command=self.start_analysis,
            style='Start.TButton'
        ).pack(
            fill='x',
            padx=(20, 20),
            pady=(20, 0)
        )

        # === Send a feedback ===
        feedback_label = tk.Label(
            self,
            text='Send Feedback',
            fg='blue',
            cursor='hand2',
            font=('TkDefaultFont', 14, 'underline'),
            bg=self.bg_color
        )
        feedback_label.pack(pady=(15, 20))
        feedback_label.bind('<Button-1>', self.open_feedback_email)


    def _build_path_selector(self, parent, label_text, command, **pack_kwargs):
        frame = ttk.Frame(parent)
        frame.pack(**pack_kwargs)

        ttk.Label(frame, text=label_text).pack(anchor='w')

        var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=var)
        entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

        button = ttk.Button(frame, text='Browse', command=command)
        button.pack(side='left')

        return var
    
    def _build_checkbutton(self, parent, val, text, **pack_kwargs):
            var = tk.BooleanVar(value=val)
            ttk.Checkbutton(
                parent,
                text=text,
                variable=var
            ).pack(
                **pack_kwargs
            )

            return var
    
    def on_close(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            confirm = messagebox.askyesno(
                "Confirm Exit",
                "An analysis is currently running.\nAre you sure you want to exit?"
            )
            if not confirm:
                return
        
        self.stop_event.set()
        self.destroy()

    def select_files(self):
        patterns = ' '.join(f'*{ext}' for ext in INPUT_EXT)
        paths = filedialog.askopenfilenames(
            title='Select input images',
            filetypes=(
                ('Images', patterns),
                ('All Files', '*.*')
            )
        )
        if paths:
            all_paths = ', '.join(paths)
            self.input_file_path_var.set(all_paths)

    def select_directory(self):
        path = filedialog.askdirectory(
            title='Select output directory'
        )
        if path:
            self.output_root_dir_var.set(path)

    def open_advanced_preferences(self):
        if not self.advanced_window or not self.advanced_window.winfo_exists():
            self.advanced_window = AdvancedPreferencesWindow(self, self.params_mgr)
            self.advanced_window.protocol('WM_DELETE_WINDOW', self.close_advanced_preferences)
        else:
            self.advanced_window.lift()
            self.advanced_window.focus_force()

    def close_advanced_preferences(self):
        if self.advanced_window is not None:
            self.advanced_window.destroy()
            self.advanced_window = None

    def start_analysis(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning(
                'Analysis in Progress',
                'An analysis is already running. Please wait for it to complete.'
            )
            return
        
        self.analysis_thread = threading.Thread(target=self._start_analysis)
        self.analysis_thread.start()

    def _start_analysis(self): 
        img_paths = self.input_file_path_var.get()
        output_root_dir = self.output_root_dir_var.get()

        if not (img_paths and output_root_dir):
            self.after(0, lambda: messagebox.showwarning(
                'Missing Path',
                'Please select at least one input image and an output directory.'
            ))
            return
        
        # Check that at least one output option is specified
        save_flags = [
            self.save_graph.get(),
            self.save_mask.get(),
            self.save_report.get(),
            self.save_segmentation_rois.get(),
            self.save_bbox_rois.get(),
            self.save_lin_decomp_rois.get()
        ]
        if not any(save_flags):
            self.after(0, lambda: messagebox.showwarning(
                'Invalid Selection',
                'You must select at least one output option.'
            ))
            return
        
        self.after(0, lambda: messagebox.showinfo(
            'Status Information',
            'Analysis Started'
        ))
        
        dna2graph = DNA2Graph(
            output_root_dir=output_root_dir,
            config=self.params_mgr.in_memory_config,
            trained_seg=self.trained_seg.get(),
            save_graph=self.save_graph.get(),
            save_mask=self.save_mask.get(),
            save_report=self.save_report.get(),
            save_segmentation_rois=self.save_segmentation_rois.get(),
            save_bbox_rois=self.save_bbox_rois.get(),
            save_lin_decomp_rois=self.save_lin_decomp_rois.get(),
            clean_cache=self.clean_cache.get(),
            stop_event=self.stop_event
        )

        img_paths = img_paths.split(', ')
        for img_path in tqdm(img_paths, desc='Processing images', unit='image'):
            try:
                dna2graph.forward(img_path)
            except UserCancelledError:
                return
            except Exception as e:
                self.after(0, lambda path=img_path, err=str(e): messagebox.showinfo(
                    'Image Processing Error',
                    f'An error occurred while processing image:\n{path}\n{err}\n'
                    f'Skipping to the next image.'
                ))
                
        self.after(0, lambda: messagebox.showinfo(
            'Status Information',
            'Analysis Completed'
        ))

    def open_feedback_email(self, event=None):
        webbrowser.open(
            f'mailto:{DEVELOPER_EMAIL}?subject=Feedback {APP_NAME}'
        )