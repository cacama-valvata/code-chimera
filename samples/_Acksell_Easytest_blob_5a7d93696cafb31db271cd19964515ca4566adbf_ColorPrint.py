import sys

"""
Color code guide: https://bixense.com/clicolors/
and https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit

Modified class from https://stackoverflow.com/a/47622205/9405408
"""

# Colored printing functions for strings that use universal ANSI escape sequences.
# fail: bold red, pass: bold green, warn: bold yellow, 
# info: bold blue, bold: bold white

class ColorPrint:
    @staticmethod
    def print(message, foreground=231, background=0, end = '\n'):
        """
        For color codes: https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
        """
        prepend=''
        if background not in range(0,256): raise IndexError("Background must be between 0-255")
        elif background:
            prepend='\x1b[48;5;{}m'.format(background)
        ansi_code = '{}\x1b[38;5;{}m'.format(prepend, foreground)
        sys.stderr.write(ansi_code + message + '\x1b[0m' + end)

    @staticmethod
    def fail(message, end = '\n', background=0, foreground=9):
        if background: 
            background=9
            foreground=0
        ColorPrint.print(message, foreground=foreground, background=background, end=end)

    @staticmethod
    def green(message, end = '\n', background=0, foreground=48):
        if background: 
            background=48
            foreground=0
        ColorPrint.print(message, background=background, foreground=foreground, end=end)
        
    @staticmethod
    def warn(message, end = '\n', background=0, foreground=3):
        if background: 
            background=3
            foreground=0
        ColorPrint.print(message, background=background, foreground=foreground, end=end)

    @staticmethod
    def info(message, end = '\n', background=0, foreground=6):
        if background: 
            background=6
            foreground=0
        ColorPrint.print(message, background=background, foreground=foreground, end=end)
        
    @staticmethod
    def white(message, end = '\n', background=0, foreground=231):
        if background: 
            background=7
            foreground=0
        ColorPrint.print(message, background=background, foreground=foreground, end=end,)

        
if __name__ == "__main__":
    ColorPrint.fail("ThisText fails: ", end="")
    ColorPrint.fail("HARD",background=1)
    ColorPrint.green("ThisText passes ",end="")
    ColorPrint.green("HARD",background=1)
    
    ColorPrint.warn("ThisText warns: ", end="")
    ColorPrint.warn("HARD",background=1)

    ColorPrint.info("ThisText informs: ", end="")
    ColorPrint.info("HARD",background=1)
    ColorPrint.white("ThisText is white ",end="")
    ColorPrint.white("HARD",background=1)

    