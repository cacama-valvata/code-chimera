#!/usr/bin/env python

import sys
import termios
import tty

from abstract2048 import Abstract2048


class Shell2048(Abstract2048):
    def __str__(self):
        return 'Score: {score}\n{m}\n'.format(**self.__dict__)

    def run(self):
        def getchar():
            """ https://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user """
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        print(self)
        while not self.game_over():
            c = getchar()
            if c == '4':
                slides = self.left()
            elif c == '8':
                slides = self.up()
            elif c == '6':
                slides = self.right()
            elif c in '25':
                slides = self.down()
            elif c == '0' and self.m.min() == 0:  # cheating
                slides = self.pop()
            elif c == 'q':
                print('Y U QUIT ? :(')
                return
            else:
                slides = None
            if slides:
                print(self)
                print(slides)
        print('game over', self)

if __name__ == '__main__':
    try:
        g = Shell2048(*sys.argv[1:])
        g.pop()
        g.run()
    except ValueError:
        print('usage: %s [height [width [goal [initial_score]]]]' % sys.argv[0])
        sys.exit(1)
