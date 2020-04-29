# coding: utf-8


# several data readers need optional dependencies. There's no
# appropriate builtin exception
class MissingDependencyException(Exception):
    pass


class DataReaderBase(object):
    """Read data from file system and yield as dicts.

    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: A number
            of DataReaders need specific additional packages.
            If any are missing, this will be raised.
    """

    @classmethod
    def from_opt(cls, opt):
        """Alternative constructor.

        Args:
            opt (argparse.Namespace): The parsed arguments.
        """

        return cls()

    @classmethod
    def _read_file(cls, path): # 用二进制读文件是因为data有可能会是语音or图像, 如果是文本后面再decode(utf8)一下就好了
        """Line-by-line read a file as bytes."""
        with open(path, "rb") as f:
            for line in f:
                yield line

    @staticmethod
    def _raise_missing_dep(*missing_deps):
        """Raise missing dep exception with standard error message."""
        raise MissingDependencyException(
            "Could not create reader. Be sure to install "
            "the following dependencies: " + ", ".join(missing_deps))

    def read(self, data, side, src_dir):
        """Read data from file system and yield as dicts."""
        raise NotImplementedError()
