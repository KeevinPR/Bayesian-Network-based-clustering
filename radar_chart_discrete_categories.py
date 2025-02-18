import numpy as np
import textwrap
#Este código es una variacion de radar_chart_discrete donde a partir de una codificacion con nº enteros de las categorias de las distintas variables
#conseguimos un radar chart que muestra los valores que toma cada representante de cada cluster
class ComplexRadar():
    """
    Create a complex radar chart with different scales for each variable
    Parameters
    ----------
    fig : figure object
        A matplotlib figure object to add the axes on
    variables : list
        A list of variables
    ranges : list
        A list of tuples (min, max) for each variable
    n_ring_levels: int, defaults to 5
        Number of ordinate or ring levels to draw
    show_scales: bool, defaults to True
        Indicates if we the ranges for each variable are plotted
    """



    def __init__(self, fig, df_categories, show_scales=True):
        # Calculate angles and create for each variable an axes
        # Consider here the trick with having the first axes element twice (len+1)
        angles = np.arange(0, 360, 360. / len(df_categories.keys()))
        #axes = [fig.add_axes([0.3, 0.4, 0.4,0.4], polar=True, label="axes{}".format(i)) for i in
        axes = [fig.add_axes([0.2, 0.3, 0.7, 0.7], polar=True, label="axes{}".format(i)) for i in        
                range(len(df_categories.keys()) +1)]

        # Ensure clockwise rotation (first variable at the top N)
        for ax in axes:
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)

        # Writing the ranges on each axes
        baseline=len(df_categories[list(df_categories.keys())[0]].values())+2
        for i, ax in enumerate(axes):
            # Here we do the trick by repeating the first iteration
            j = 0 if (i == 0 or i == 1) else i - 1
            cat=df_categories[list(df_categories.keys())[j]]
            ax.set_ylim(0,len(cat.values())+1)
            # Set endpoint to True if you like to have values right before the last circle
            grid = np.linspace(0,len(cat.values())+1, num=len(cat.values())+1,
                               endpoint=False)
            gridlabel =['']  # remove values from the center
            gridlabel.extend(cat.keys())

            lines, labels = ax.set_rgrids(grid, labels=gridlabel, angle=angles[j])

            ax.spines["polar"].set_visible(False)
            ax.grid(visible=False)

            if show_scales == False:
                ax.set_yticklabels([])

        # Set all axes except the first one unvisible
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)

        # Setting the attributes
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ax = axes[0]
        self.ax1 = axes[1]
        self.plot_counter = 0

        # Draw (inner) circles and lines
        #self.ax.yaxis.grid()
        self.ax.xaxis.grid()

        # Draw outer circle
        self.ax.spines['polar'].set_visible(True)

        # ax1 is the duplicate of axes[0] (self.ax)
        # Remove everything from ax1 except the plot itself
        self.ax1.axis('off')
        self.ax1.set_zorder(9)

        # Create the outer labels for each variable
        l, text = self.ax.set_thetagrids(angles, labels=df_categories.keys())

        # Beautify them
        labels = [t.get_text() for t in self.ax.get_xticklabels()]
        labels = ['\n'.join(textwrap.wrap(l, 15,break_long_words=False)) for l in labels]
        self.ax.set_xticklabels(labels)

        for t, a in zip(self.ax.get_xticklabels(), angles):
            if a == 0:
                t.set_ha('center')
            elif a > 0 and a < 180:
                t.set_ha('left')
            elif a == 180:
                t.set_ha('center')
            else:
                t.set_ha('right')

        self.ax.tick_params(axis='both', pad=15)

    def _scale_data(self, data, df_categories):

        baseline = len(df_categories[list(df_categories.keys())[0]].values()) + 2
        sdata=[]
        for var in df_categories.keys():
            sdata.append(df_categories[var][data[var]]/(len(list(df_categories[var].values()))+2) * (baseline))

        return sdata

    def plot(self, data, df_categories, *args, **kwargs):
        """Plots a line"""
        sdata = self._scale_data(data, df_categories)
        self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
        self.plot_counter = self.plot_counter + 1

    def fill(self, data, df_categories,*args, **kwargs):
        """Plots an area"""
        sdata = self._scale_data(data, df_categories)
        sdata.append(sdata[0])
        self.ax1.fill(self.angle, sdata, *args, **kwargs)

    def use_legend(self, *args, **kwargs):
        """Shows a legend"""
        self.ax1.legend(*args, **kwargs)

    def set_title(self, title, pad=25, **kwargs):
        """Set a title"""
        self.ax.set_title(title, pad=pad, **kwargs)