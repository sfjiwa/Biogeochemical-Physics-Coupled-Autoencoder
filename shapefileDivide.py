
import os
import sys

import tkinter
from tkinter import Tk
import tkinter.filedialog as tkfd
import tkinter.simpledialog as tksd

import fiona


class simpleform_ap(Tk):

    def __init__(self, parent, options):
        Tk.__init__(self, parent)
        self.initialize(options)
        self.grid()
        self.field = None

    def initialize(self, options):
        # Dropdown Menu
        self.dropVar = tksd.StringVar()
        self.dropVar.set(options[0]) # default choice is first field name
        self.dropMenu1 = tksd.OptionMenu(self, self.dropVar, *options, command=self.func)
        self.dropMenu1.grid(column=1,row=2)
        self.CloseButton = tksd.Button(self, text = "Start splitting", command = self.close_window)
        self.CloseButton.grid(column=1, row=3)

    def func(self,value):
        self.field = value

    def close_window(self):
        self.destroy()

def create_field_form(options):
    form = simpleform_ap(None, options)
    form.title('Select the field to split the shapefile')
    form.mainloop()
    print("Field used for splitting is", form.field)
    return(form.field)


def gui_inputs():
    """
    """
    # locate your input shapefile
    infile = tkfd.askopenfilename(title = "Select your shapefile", 
                                initialdir = "C:/", 
                                filetypes = (("shapefile", "*.shp"), 
                                             ("all files", "*.*")))

    # define destination -folder- where output -shapefiles- will be written
    dest = tkfd.askdirectory(title = "Select your output folder", 
                            initialdir = "C:/")


    # choose field to split by attributes
    field_options = _extract_attribute_field_names(infile)
    field = create_field_form(field_options)

    return infile, field, dest

def _extract_attribute_field_names(infile):
    """extract the field names from a fiona compatible vector file
    """
    with fiona.open(infile) as source:
        field_names = source.meta["schema"]["properties"].keys()

    return list(field_names)


def split_shape_by_attribute(infile, field, dest):
    """split by attributes for shapefiles
    Works fine, for both unique and non-unique attribute values! Records with the 
    same attributes are grouped in seperate, new shapefiles...
    Parameters
    ----------
    infile : str
        filename of the input shapefile
    field : str
        name of the atrribute table field
    dest : str
        define destination -folder- where output -shapefiles- 
        will be written
    """
    with fiona.open(infile) as source:
        meta = source.meta
        for f in source:
            outfile = os.path.join(dest, "%s.shp" % f['properties'][field])
            try:
                with fiona.open(outfile, 'a', **meta) as sink:
                    sink.write(f)
            except:
                with fiona.open(outfile, 'w', **meta) as sink:
                    sink.write(f)

def main(argv=None):
    """
    """     

    infile, field, dest = gui_inputs()
    
    print("Start splitting procedure...")
    split_shape_by_attribute(infile, field, dest)
    print("... splitting completed!")


if __name__ == "__main__":
    sys.exit(main())