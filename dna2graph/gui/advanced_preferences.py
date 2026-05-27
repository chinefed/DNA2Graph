import tkinter as tk
from tkinter import ttk

from dna2graph.constants import (
    TRAINED_SUFFIX,
    SECTIONS_BASE,
    SECTIONS_TRAINED
)


class AdvancedPreferencesWindow(tk.Toplevel):
    def __init__(self, parent, params_mgr):
        super().__init__(parent)
        self.title('Advanced Preferences')
        self.geometry('470x650')
        self.resizable(False, True)

        self.tk_vars = {}
        self.params_mgr = params_mgr
        self.parent = parent
        self.trained_seg = self.parent.trained_seg.get()

        self.configure(bg=self.parent.bg_color)
        self._build_ui()

    def _build_ui(self):
        # Create scrollable frame
        container = ttk.Frame(self)
        container.pack(
            fill='both',
            expand=True,
            padx=10,
            pady=10
        )

        canvas = tk.Canvas(
            container,
            highlightthickness=0,
            bg=self.parent.bg_color
        )
        canvas.pack(
            side='left',
            fill='both',
            expand=True
        )

        scrollbar = ttk.Scrollbar(
            container,
            orient='vertical',
            command=canvas.yview
        )
        scrollbar.pack(
            side='right',
            fill='y'
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        scroll_frame = ttk.Frame(canvas)

        canvas.create_window(
            (0, 0),
            window=scroll_frame,
            anchor='nw'
        )

        scroll_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )

        # Create sections
        sections = SECTIONS_TRAINED if self.trained_seg else SECTIONS_BASE
        for sec_name in sections:
            label_text = sec_name.replace(TRAINED_SUFFIX, '').replace('_', ' ')
            ttk.Label(
                scroll_frame,
                text=label_text.capitalize(),
                font=('Arial', 14, 'bold')
            ).pack(
                pady=(10, 5)
            )

            # Add parameter entries
            self.tk_vars[sec_name] = {}

            sec_params = self.params_mgr.in_memory_config[sec_name]
            for k, v in sec_params.items():
                self.tk_vars[sec_name][k] = self._make_param_entry(
                    scroll_frame,
                    k,
                    v
                )

        # Save as default checkbox
        save_default_frame = ttk.Frame(self)
        save_default_frame.pack(fill='x', pady=(5, 0))

        self.save_as_default_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            save_default_frame,
            text='Save as default config',
            variable=self.save_as_default_var
        ).pack(
            side='left',
            padx=12,
            pady=2
        )

        # Save, Cancel, Reset buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', pady=10)

        ttk.Button(
            btn_frame,
            text='Save',
            command=self.save
        ).pack(
            side='left',
            padx=10
        )

        ttk.Button(
            btn_frame,
            text='Cancel',
            command=self.destroy
        ).pack(
            side='left',
            padx=10
        )

        ttk.Button(
            btn_frame,
            text='Reset to defaults',
            command=self.reset_to_defaults
        ).pack(
            side='left',
            padx=10
        )

    def _make_param_entry(self, parent, key, val):
        '''
        Creates a parameter entry widget based on the type of the value.
        '''
        # Only support certain types
        supported_types = (bool, int, float, str)
        val_type = type(val)
        if val_type not in supported_types:
            raise ValueError(
                f"Unsupported parameter type for '{key}': {val_type}")

        # Create row frame
        row = ttk.Frame(parent)
        row.pack(fill='x', padx=10, pady=2)

        # Label (formatting key for readability)
        label_text = key.replace('_', ' ').capitalize() + ':'
        ttk.Label(
            row,
            text=label_text,
            width=24,
            anchor='w',
        ).pack(side='left')

        # Mapping of type to Tk variable class
        var_map = {
            bool: tk.BooleanVar,
            int: tk.IntVar,
            float: tk.DoubleVar,
            str: tk.StringVar,
        }
        var = var_map[val_type](value=val)

        # Widget selection
        if val_type is bool:
            # Use Checkbutton for bool
            ttk.Checkbutton(row, variable=var).pack(side='left')
        else:
            # Use Entry for int, float, str
            ttk.Entry(row, textvariable=var, width=15).pack(side='left')

        return var

    def save(self):
        updates = {}
        for sec_name, sec_params in self.tk_vars.items():
            updates[sec_name] = {}
            for k, var in sec_params.items():
                updates[sec_name][k] = var.get()

        # Update session parameters in memory
        self.params_mgr.update_in_memory(updates)

        # Save to user config if requested
        if self.save_as_default_var.get():
            self.params_mgr.save_user_config()

        self.parent.close_advanced_preferences()

    def reset_to_defaults(self):
        for sec_name, sec_params in self.tk_vars.items():
            for k, var in sec_params.items():
                var.set(self.params_mgr.default_config[sec_name][k])
