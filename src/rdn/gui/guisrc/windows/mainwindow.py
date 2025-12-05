import enum
import tkinter as tk

from .plotwindow import PlotWindow

class MainWindow(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.model_p_dict = []
        self.input_entries = []

        # Left Panel
        self._draw_left_panel()

        # For now I skip drawing the right panel, I use the default
        # visualization in Simulate
        # Right Panel (for the plot)
        # self._draw_right_panel()
        


    def _draw_left_panel(self):
        frm_l = tk.Frame(master=self,
                         width=30,
                         height=50)
        
        # First three inputs
        for i, (label_text, default_value) \
                in enumerate(zip(['Simulation time [min] \u207D\u00B9\u207E',
                                  'Inter-spine distance [um] \u207D\u00B2\u207E',
                                  'Spine number \u207D\u00B3\u207E'],
                                 [40,1,100])):
            frame = tk.Frame(master=frm_l)
            tk.Label(master=frame, text=label_text, width=25, anchor='w').pack(side=tk.LEFT)
            entry = tk.Entry(master=frame, width=10)
            entry.insert(0,str(default_value))
            entry.pack(side=tk.LEFT)
            self.input_entries.append(entry)
            frame.grid(row=i, column=0)

        # Stimulus scheme description

        # Stimulus scheme
        frame = tk.Frame(master=frm_l)
        frame.grid(row=4, column=0, pady=(10,0))

        tk.Label(master=frame, 
                 text='Stimulus scheme \u207D\u2074\u207B\u2075\u207B\u2076\u207E',
                 width=25, 
                 anchor='w').grid(row=0, column=0,sticky='w')
        tk.Label(master=frame, 
                 text=' Logic: start, number, stride',
                 font='Helvetica 7 italic',
                 anchor='w').grid(row=1,
                                  column=0,
                                  sticky='w')

        # We append all the entries to self
        for col, default_value in enumerate([1,2,2]):
            entry = tk.Entry(master=frame, width=3,)
            entry.insert(0, str(default_value))
            entry.grid(row=0, column=col+1, sticky='ns', rowspan=2)
            self.input_entries.append(entry)


        # We add the dynamical parameters one by one. Very stupid but I don't
        # have a better idea for now. Start with the general frame
        # Stimulus scheme description
        frame = tk.Frame(master=frm_l)
        frame.grid(row=5, column=0, pady=(20,0))
        tk.Label(master=frame, 
                 text="Dynamics' parameters",
                 font='Helvetica 8 italic',
                 anchor='w').grid(sticky='we',
                                  row=3,
                                  column=0)


        frame = tk.Frame(master=frm_l)
        frame.grid(row=6, column=0, pady=(10,0))

        # Chi 
        tk.Label(master=frame, 
                text='chi \u207D\u2077\u207E',
                width=10, 
                anchor='w').grid(row=0, column=0)

        entry = tk.Entry(master=frame, width=10)
        entry.insert(0, '1')
        entry.grid(row=0, column=1)
        self.input_entries.append(entry)

        
        # Pi
        tk.Label(master=frame, 
                text='Pi \u207D\u2078\u207E',
                width=10, 
                anchor='w').grid(row=0, column=2, padx=(20,0))

        entry = tk.Entry(master=frame, width=10)
        entry.insert(0, '61603376')
        entry.grid(row=0, column=3)
        self.input_entries.append(entry)


        # Spine specific stuff
        frame = tk.Frame(master=frm_l)
        frame.grid(row=7, column=0, pady=(10,0))


        # K stuff: S, sigma, tau, mu_log
    
        texts = [
            'Ks \u207D\u2079\u207E',
            'sigma_K \u207D\u00B9\u2070\u207E', 
            'tau_K \u207D\u00B9\u00B9\u207E', 
            'mu_log_K \u207D\u00B9\u00B2\u207E', 
        ]
        values = ['30000', '1', '10.48', '8.36']

        for row, (text, v) in enumerate(zip(texts, values)):
            tk.Label(master=frame, 
                    text=text,
                    width=10, 
                    anchor='w').grid(row=row, column=0)
            
            entry = tk.Entry(master=frame, width=10)
            entry.insert(0, v)
            entry.grid(row=row, column=1)
            self.input_entries.append(entry)


        # N stuff: S, sigma, tau, mu_log
        texts = [
            'Ns \u207D\u00B9\u00B3\u207E',
            'sigma_N \u207D\u00B9\u2074\u207E', 
            'tau_N \u207D\u00B9\u2075\u207E', 
            'mu_log_N \u207D\u00B9\u2076\u207E', 
        ]
        values = ['34950','1.21', '11.30', '8.99']

        for row, (text, v) in enumerate(zip(texts, values)):
            tk.Label(master=frame, 
                    text=text,
                    width=10, 
                    anchor='w').grid(row=row, column=2, padx=(20,0))
            
            entry = tk.Entry(master=frame, width=10)
            entry.insert(0, v)
            entry.grid(row=row, column=3)
            self.input_entries.append(entry)

        

        # Spine specific stuff
        frame = tk.Frame(master=frm_l)
        frame.grid(row=8, column=0, pady=(13,0))

        texts = ['var_log_K \u207D\u00B9\u2077\u207E',
                 'var_log_K \u207D\u00B9\u2078\u207E',
                 'var_log_K \u207D\u00B9\u2079\u207E',]
        
        values = ['0.27', '0.20', '0.25',]

        for text, v in zip(texts, values):
            tk.Label(master=frame, 
                    text=text,
                    width=12, 
                    anchor='e').pack(side=tk.LEFT, padx=(0,5), )
            
            entry = tk.Entry(master=frame, width=5)
            entry.insert(0, v)
            entry.pack(side=tk.LEFT)
            self.input_entries.append(entry)

        # Finally add the button
        tk.Button(master=frm_l,
                 text='RUN',
                 command=self._open_new_window).grid(row=9,
                                                     column=0, 
                                                     pady=(20,0,),
                                                     sticky='we')


        # Window to copypaste the pardict
        # frame = tk.Frame(master=frm_l)
        #
        # tk.Label(master=frame, 
        #         text='Pardict',
        #         width=10, 
        #         anchor='w').pack(side=tk.LEFT)
        #
        # txt_model_p_dict = tk.Text(master=frame,
        #                            width=100,
        #                            height=10)
        #
        # self.input_entries.append(txt_model_p_dict)
        # frame.grid(row=6, column=0)
        # txt_model_p_dict.pack(side=tk.LEFT)
        frm_l.pack(side=tk.LEFT)

    def _open_new_window(self):
        new_root = tk.Toplevel(self.parent)
        PlotWindow(new_root, self.input_entries).pack(side='top', 
                                                      fill='both', 
                                                      expand=True)

if __name__ == '__main__':
    root = tk.Tk()
    main_window = MainWindow(root).pack(side='top', fill='both', expand=True)
    root.mainloop()

