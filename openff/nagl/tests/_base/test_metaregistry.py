import pytest

from openff.nagl._base.metaregistry import create_registry_metaclass

class TestMetaRegistry:

    Registry = create_registry_metaclass(ignore_case=False)
    RegistryIgnoreCase = create_registry_metaclass(ignore_case=True)

    class TestClass(metaclass=Registry):
        name = "TestKey"
    
    class TestClassIgnoreCase(metaclass=RegistryIgnoreCase):
        name = "TestKey"

    def test_create_registry_metaclass(self): 
        assert "TestKey" in self.Registry.registry
        assert self.Registry.registry["TestKey"] is self.TestClass
        assert "testkey" in self.RegistryIgnoreCase.registry
        assert self.RegistryIgnoreCase.registry["testkey"] is self.TestClassIgnoreCase

    def test_key_transform(self):
        assert "TestKey" in self.Registry.registry
        assert self.Registry._get_by_key("TestKey") is self.TestClass
        with pytest.raises(KeyError):
            self.Registry._get_by_key("testkey")

    def test_key_transform_ignore_case(self):
        assert "testkey" in self.RegistryIgnoreCase.registry
        assert "TestKey" not in self.RegistryIgnoreCase.registry
        assert self.RegistryIgnoreCase._get_by_key("TestKey") is self.TestClassIgnoreCase
        assert self.RegistryIgnoreCase._get_by_key("testkey") is self.TestClassIgnoreCase
    
    def test_get_class(self):
        assert self.Registry._get_class(self.TestClass) is self.TestClass
        assert self.Registry._get_class(self.TestClass()) is self.TestClass
        assert self.Registry._get_class("TestKey") is self.TestClass
        with pytest.raises(KeyError):
            self.Registry._get_class("testkey")
    
    def test_get_object(self):
        assert isinstance(self.Registry._get_object(self.TestClass), self.TestClass)
        assert isinstance(self.Registry._get_object(self.TestClass()), self.TestClass)
        assert isinstance(self.Registry._get_object("TestKey"), self.TestClass)
        with pytest.raises(KeyError):
            self.Registry._get_object("testkey")
    
