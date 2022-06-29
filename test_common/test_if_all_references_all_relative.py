from unittest import TestCase
from tg.common.tools_bak.fix_absolute_import import get_tg_files, find_module_name
from yo_fluq_ds import Query


class RelativeImportsTestCase(TestCase):
    def test_imports_are_relative(self):
        for file in get_tg_files():
            for index, line in Query.file.text(file).feed(enumerate):
                if index == 0 and line == '#Absolute import allowed':
                    break
                module = find_module_name(line)
                if module is not None:
                    self.fail(f"Absolute import of {module} in file {file} at line {index}")
