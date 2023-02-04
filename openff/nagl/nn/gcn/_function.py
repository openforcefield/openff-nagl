def copy_u(msg: str, out: str):
    def wrapped(edges):
        return {out: edges.src[msg]}

    return wrapped


def u_mul_e(msg: str, weight: str, out: str):
    def wrapped(edges):
        return {out: edges.data[msg] * edges.data[weight]}

    return wrapped


def mean(msg: str, out: str):
    def wrapped(nodes):
        import torch

        return {out: torch.mean(nodes.mailbox[msg], dim=1)}

    return wrapped


def sum(msg: str, out: str):
    def wrapped(nodes):
        import torch

        return {out: torch.sum(nodes.mailbox[msg], dim=1)}

    return wrapped


def max(msg: str, out: str):
    def wrapped(nodes):
        import torch

        return {out: torch.max(nodes.mailbox[msg], dim=1)}

    return wrapped
