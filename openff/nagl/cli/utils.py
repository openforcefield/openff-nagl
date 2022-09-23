import functools
import os
import traceback
import tqdm
from typing import Any, Callable, List, Tuple


def as_batch_function_with_captured_errors(
    func: Callable = lambda x: None,
    desc: str = None,
) -> List[Tuple[Any, str]]:
    def wrapper(batch: List[Any]):
        results = []
        for entry in tqdm.tqdm(batch, ncols=80, desc=desc):
            results.append(
                try_and_return_error(
                    functools.partial(func, entry), error=f"Failed to process {entry}"
                )
            )
        return results

    return wrapper


def get_log_file_from_output(file: str, suffix: str = "-errors.log") -> str:
    file = str(file)
    fields = file.split("/")
    base = fields[-1].rsplit(".", maxsplit=1)[0]
    fields[-1] = base + suffix
    return "/".join(fields)


def get_default_manager(ctx):
    from openff.nagl.app.distributed import Manager

    try:
        manager = ctx.obj.get("manager")
    except AttributeError:
        manager = None

    return manager


def try_and_return_error(func, error: str = "Failed"):
    try:
        output = func()
    except (BaseException, Exception) as e:
        tb = traceback.format_exception(e, value=e, tb=e.__traceback__)
        err = f"{error}: {tb}"
        output = None
    else:
        err = None
    return output, err


def write_error_to_file_object(file_object, error: str, separator: bool = True):
    if separator:
        file_object.write("=" * 79)
    file_object.write(error + "\n")
    file_object.flush()


def preprocess_args(manager, input_file, output_file):
    from openff.nagl.app.distributed import Manager

    input_file = os.path.abspath(str(input_file))
    output_file = os.path.abspath(str(output_file))
    log_file = get_log_file_from_output(output_file)

    if manager is None:
        manager = Manager()

    return manager, input_file, output_file, log_file
