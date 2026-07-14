'''Console view helpers for user-facing status output.'''

def show(message=''):
    '''Print a user-facing status message.'''
    print(message)

def show_section(title):
    '''Print a standard section heading.'''
    print('')
    print('=' * 50)
    print(title)
    print('=' * 50)
