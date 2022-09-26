from typing import ClassVar, Dict, Type, Any


class MetaRegistryMixin(type):
    registry: ClassVar[Dict[str, Type]] = {}

    @classmethod
    def _key_transform(cls, key):
        return key

    @classmethod
    def _get_by_key(cls, key: str):
        try:
            return cls.registry[cls._key_transform(key)]
        except KeyError:
            raise KeyError(
                f"Unknown {cls.__name__} type: {key}. "
                f"Supported types: {list(cls.registry.keys())}"
            )
    

    @classmethod
    def _get_class(cls, obj: Any):
        if isinstance(obj, cls):
            return obj
        if isinstance(type(obj), cls):
            return type(obj)
        return cls._get_by_key(obj)


    @classmethod
    def _get_object(cls, obj: Any):
        if isinstance(type(obj), cls):
            return obj
        obj = cls._get_class(obj)
        return obj()


class MetaRegistryMixinAnyCase(MetaRegistryMixin):
    @classmethod
    def _key_transform(cls, key):
        return key.lower()


def create_registry_metaclass(name_attribute: str = "name", ignore_case: bool = True):
    base_class = MetaRegistryMixinAnyCase if ignore_case else MetaRegistryMixin

    class RegistryMeta(base_class):
        registry: ClassVar[Dict[str, Type]] = {}
        _key_attribute: ClassVar[str] = name_attribute

        def __init__(cls, name, bases, namespace, **kwargs):
            super().__init__(name, bases, namespace, **kwargs)
            if hasattr(cls, cls._key_attribute):
                attr_val = getattr(cls, cls._key_attribute)
                if not isinstance(attr_val, property) and attr_val:
                    cls.registry[cls._key_transform(attr_val)] = cls

    return RegistryMeta

