major = 1
minor = 0
micro = None

pre_release = ".alpha"
post_release = None
dev_release = None

__version__ = '{}'.format(major)

if minor is not None:
    __version__ += '.{}'.format(minor)

if micro is not None:
    __version__ += '.{}'.format(micro)

if pre_release is not None:
    __version__ += '{}'.format(pre_release)

if post_release is not None:
    __version__ += '.post{}'.format(post_release)

if dev_release is not None:
    __version__ += '.dev{}'.format(dev_release)
