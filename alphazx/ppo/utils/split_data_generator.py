from typing import Any


def split_data_generator(data: dict[str, list[Any]], batch_size: int, shuffle: bool) -> list[dict[str, Any]]:
    """
    :param data:
    :param batch_size:
    :param shuffle:
    :return: Dictionary containing entries for keys: 'logit', 'action', 'value', 'return', 'weight'
    """
    raise NotImplementedError
