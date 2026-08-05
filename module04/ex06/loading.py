import time
from math import floor


def ft_progress(lst):
    """function to display an progress bar,
    give behave like an iterator over a iterator lst.
    """
    start_time = time.time()
    lenght = len(lst)
    percentage = 0
    eta = 0
    vide = ' '
    for i, el in enumerate(lst):
        yield el
        schema = ''
        percentage = floor((i + 1) * 100 / lenght)
        schema = "=" * floor(percentage / 2)
        if percentage != 100:
            schema += '>'
        schema += ' ' * floor((100 - percentage - 1) / 2)
        curr_time = time.time() - start_time
        vitesse = (i + 1) / curr_time
        eta = (lenght - i - 1) / vitesse
        bar = (
            f'ETA: {eta:.2f}s [{percentage}%]' +
            f'[{schema}] {i + 1}/{lenght} | elapsed time ' +
            f'{curr_time:.2f}s'
        )
        print(vide, end='\r')
        print(f"{bar}", end='\r')
        vide = ' ' * len(bar)
    print()


if __name__ == '__main__':
    listy = range(3333)
    ret = 0
    for elem in ft_progress(listy):
        ret += elem
        time.sleep(0.005)
    print()
    print(ret)
