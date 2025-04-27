import re
import sys
import os


class FortranParser:
    """
    静态分析引擎组件，用于解析通用Fortran代码
    """

    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        self.ast = {}
        self.loops = []
        self.variables = {}
        self.declarations = {}
        self.arrays = {}

    def lexical_analysis(self):
        """
        对Fortran源代码进行词法分析
        """
        print("执行词法分析...")

        # 删除注释并处理续行符
        clean_lines = []
        continued_line = ""

        for line in self.source_code.split('\n'):
            # 处理注释
            if '!' in line:
                line = line.split('!')[0]

            line = line.strip()

            # 处理续行符
            if line and line.endswith('&'):
                continued_line += line[:-1].strip() + " "
            elif continued_line:
                continued_line += line
                if continued_line.strip():
                    clean_lines.append(continued_line)
                continued_line = ""
            elif line:
                clean_lines.append(line)

        # 标记化代码
        for line in clean_lines:
            # 分割保留字符串
            tokens = re.findall(r'[^"\s]+|"[^"]*"', line)
            self.tokens.extend(tokens)

        return self.tokens

    def detect_declarations(self):
        """
        检测变量声明
        """
        print("检测变量声明...")

        declarations = {}
        arrays = {}
        types = ['integer', 'real', 'double', 'complex', 'logical', 'character']

        for line in self.source_code.split('\n'):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('!'):
                continue

            # 检查变量声明
            line_lower = line.lower()
            if any(line_lower.startswith(type_name) for type_name in types):
                # 提取类型
                type_match = re.match(r'(\w+)(?:\([\w*,:]+\))?\s*::', line_lower)
                var_type = type_match.group(1) if type_match else line_lower.split()[0]

                # 提取变量名
                if '::' in line:
                    vars_part = line.split('::')[1].strip()
                else:
                    # 对于没有::的旧式Fortran声明
                    vars_part = ' '.join(line.split()[1:])

                # 处理多个变量声明
                vars_list = vars_part.split(',')
                for var_decl in vars_list:
                    var_decl = var_decl.strip()
                    if not var_decl:
                        continue

                    # 检查是否是数组
                    array_match = re.match(r'(\w+)\s*\((.*?)\)', var_decl)
                    if array_match:
                        var_name = array_match.group(1)
                        dimensions = array_match.group(2).split(',')
                        arrays[var_name] = {
                            'type': var_type,
                            'dimensions': dimensions,
                            'read': False,
                            'write': False
                        }
                    else:
                        # 提取变量名（可能带有初始化）
                        var_name = var_decl.split('=')[0].strip()
                        declarations[var_name] = {
                            'type': var_type,
                            'read': False,
                            'write': False
                        }

        self.declarations = declarations
        self.arrays = arrays
        return declarations, arrays

    def syntax_parsing(self):
        """
        解析Fortran代码的语法
        """
        print("解析语法...")

        # 查找循环
        loops = []
        in_loop = False
        current_loop = {}
        loop_body = []

        # 不同类型的DO循环模式
        loop_patterns = [
            r'do\s+(\w+)\s*=\s*([^,]+)\s*,\s*([^,]+)(?:,\s*([^,\s]+))?',  # 标准 do i=1,n[,step]
            r'do\s+while\s*\((.*?)\)',  # do while(condition)
            r'do'  # 无条件do (需要enddo结束)
        ]

        for i, line in enumerate(self.source_code.split('\n')):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('!'):
                continue

            line_lower = line.lower()

            # 检测循环开始
            loop_start = False
            for pattern in loop_patterns:
                loop_match = re.search(pattern, line_lower)
                if loop_match:
                    loop_start = True
                    in_loop = True

                    # 根据匹配的模式创建循环信息
                    if pattern == loop_patterns[0]:  # 标准do循环
                        current_loop = {
                            'type': 'standard',
                            'index': loop_match.group(1),
                            'start': loop_match.group(2),
                            'end': loop_match.group(3),
                            'step': loop_match.group(4) if loop_match.group(4) else '1',
                            'line_number': i,
                            'body': []
                        }
                    elif pattern == loop_patterns[1]:  # do while循环
                        current_loop = {
                            'type': 'while',
                            'condition': loop_match.group(1),
                            'line_number': i,
                            'body': []
                        }
                    else:  # 简单do循环
                        current_loop = {
                            'type': 'simple',
                            'line_number': i,
                            'body': []
                        }
                    break

            # 如果是循环开始，则继续处理下一行
            if loop_start:
                continue

            # 如果在循环内，添加行到循环体或检测循环结束
            if in_loop:
                if line_lower.startswith('enddo') or line_lower == 'end do':
                    in_loop = False
                    current_loop['end_line'] = i
                    loops.append(current_loop)
                    current_loop = {}
                else:
                    current_loop['body'].append(line)

        self.loops = loops
        return loops

    def ast_construction(self):
        """
        构造抽象语法树(AST)表示
        """
        print("构造AST...")

        # 简单的AST构建，重点关注嵌套循环
        self.ast = {
            'type': 'program',
            'body': []
        }

        # 根据行号排序循环
        sorted_loops = sorted(self.loops, key=lambda x: x['line_number'])

        # 构建循环嵌套结构
        loop_stack = []

        for loop in sorted_loops:
            # 弹出已结束的循环
            while loop_stack and loop_stack[-1]['end_line'] < loop['line_number']:
                loop_stack.pop()

            # 将当前循环添加到适当的位置
            if not loop_stack:
                self.ast['body'].append(loop)
            else:
                parent = loop_stack[-1]
                if 'children' not in parent:
                    parent['children'] = []
                parent['children'].append(loop)

            loop_stack.append(loop)

        return self.ast

    def dependency_analysis(self):
        """
        分析数据依赖关系
        """
        print("分析依赖关系...")

        # 合并声明的变量和数组
        variables = {**self.declarations, **self.arrays}

        # 分析循环中的变量使用
        for loop in self.loops:
            for line in loop['body']:
                # 检查变量读写
                for var_name in variables:
                    if var_name in line:
                        # 检查变量是否被写入
                        assignment_match = re.search(r'(\w+)\s*=', line)
                        if assignment_match and assignment_match.group(1) == var_name:
                            variables[var_name]['write'] = True
                        else:
                            variables[var_name]['read'] = True

                        # 检查数组访问模式
                        if var_name in self.arrays:
                            array_accesses = re.findall(rf'{var_name}\s*\((.*?)\)', line)
                            for access in array_accesses:
                                indices = [idx.strip() for idx in access.split(',')]
                                if 'accesses' not in variables[var_name]:
                                    variables[var_name]['accesses'] = []
                                variables[var_name]['accesses'].append(indices)

        self.variables = variables
        return variables

    def analyze_loop_parallelism(self):
        """
        分析循环的并行性
        """
        print("分析循环并行性...")

        parallelizable_loops = []

        # 检查每个标准循环的并行性
        for loop in self.loops:
            if loop['type'] != 'standard':
                continue

            # 默认假设循环可并行
            is_parallelizable = True
            loop_var = loop['index']

            # 检查循环变量是否在循环体中被修改
            for line in loop['body']:
                if re.search(rf'{loop_var}\s*=', line):
                    is_parallelizable = False
                    break

            # 检查循环归纳变量
            induction_vars = {}

            # 检查数组访问中的依赖
            for var_name, var_info in self.variables.items():
                if var_info.get('write', False) and 'accesses' in var_info:
                    for access in var_info['accesses']:
                        # 如果循环索引出现在写入的数组访问中，检查可能的依赖
                        if loop_var in access:
                            # 简单检查：如果相同数组同时被读写且索引涉及循环变量，可能有依赖
                            if var_info.get('read', False):
                                is_parallelizable = False
                                break

            if is_parallelizable:
                parallelizable_loops.append(loop)

        return parallelizable_loops


class CUDAGenerator:
    """
    CUDA代码生成器，用于将Fortran转换为CUDA
    """

    def __init__(self, parser):
        self.parser = parser
        self.parallelizable_loops = []
        self.cuda_code = ""
        self.kernel_name = "fortran_kernel"

    def analyze_parallelism(self):
        """
        分析代码的并行性
        """
        self.parallelizable_loops = self.parser.analyze_loop_parallelism()
        return self.parallelizable_loops

    def generate_kernel_parameters(self):
        """
        生成CUDA内核参数
        """
        params = []

        # 添加变量作为参数
        for var_name, var_info in self.parser.variables.items():
            param_type = var_info['type']

            if var_name in self.parser.arrays:
                # 数组参数
                dimensions = var_info['dimensions']
                dim_str = ", ".join(f"dimension({d})" for d in dimensions)
                params.append(f"{param_type}, {dim_str}, device :: {var_name}")
            else:
                # 标量参数
                params.append(f"{param_type}, value :: {var_name}")

        return params

    def map_loops_to_threads(self):
        """
        将循环映射到CUDA线程
        """
        # 最多取前三个可并行循环进行映射
        mapped_loops = self.parallelizable_loops[:3]
        mapping_code = []
        bounds_check = []

        # 线程索引映射
        dims = ['x', 'y', 'z']

        for i, loop in enumerate(mapped_loops):
            if i >= 3:  # CUDA最多3D线程块
                break

            loop_var = loop['index']
            dim = dims[i]

            # 生成线程索引映射
            mapping_code.append(f"  {loop_var} = (blockIdx%{dim} - 1) * blockDim%{dim} + threadIdx%{dim}")

            # 生成边界检查
            bounds_check.append(f"{loop_var} <= {loop['end']}")

        return {
            'mapping': mapping_code,
            'bounds_check': bounds_check,
            'mapped_loops': mapped_loops
        }

    def generate_cuda_code(self):
        """
        生成CUDA代码
        """
        print("生成CUDA代码...")

        # 分析并行性
        self.analyze_parallelism()

        # 如果没有可并行化的循环
        if not self.parallelizable_loops:
            print("警告: 未找到可并行化的循环!")
            return "! No parallelizable loops found"

        # 生成内核参数
        kernel_params = self.generate_kernel_parameters()

        # 映射循环到线程
        mapping_info = self.map_loops_to_threads()
        mapping_code = mapping_info['mapping']
        bounds_check = mapping_info['bounds_check']
        mapped_loops = mapping_info['mapped_loops']

        # 开始构建CUDA代码
        self.cuda_code = f"""MODULE cuda_module
  USE cudafor
  IMPLICIT NONE

CONTAINS
  ATTRIBUTES(GLOBAL) SUBROUTINE {self.kernel_name}({', '.join(var for var in self.parser.variables)})
    ! 变量声明
"""

        # 添加变量声明
        for var_name, var_info in self.parser.variables.items():
            var_type = var_info['type']
            if var_name in self.parser.arrays:
                dimensions = var_info['dimensions']
                dim_str = ", ".join(f"dimension({d})" for d in dimensions)
                self.cuda_code += f"    {var_type}, {dim_str}, device :: {var_name}\n"
            else:
                self.cuda_code += f"    {var_type}, value :: {var_name}\n"

        # 添加线程索引映射
        self.cuda_code += "\n    ! 线程索引映射\n"
        self.cuda_code += "\n".join(mapping_code)

        # 添加边界检查
        if bounds_check:
            self.cuda_code += f"\n\n    ! 边界检查\n    IF ({' .AND. '.join(bounds_check)}) THEN\n"

        # 处理未映射的循环 (保持串行)
        remaining_loops = [loop for loop in self.parser.loops if loop not in mapped_loops]

        # 排序剩余循环
        remaining_loops.sort(key=lambda x: x['line_number'])

        # 添加循环体
        if remaining_loops:
            self.cuda_code += "\n      ! 保持串行的循环\n"

            # 递归生成嵌套循环
            def generate_loop_code(loops, indent_level=6):
                indent = " " * indent_level
                code = ""

                for loop in loops:
                    if loop['type'] == 'standard':
                        code += f"{indent}DO {loop['index']} = {loop['start']}, {loop['end']}"
                        if loop['step'] != '1':
                            code += f", {loop['step']}"
                        code += "\n"

                        # 添加循环体
                        for line in loop['body']:
                            if not any(line.strip().lower().startswith(f"do {l['index']}") for l in loops):
                                code += f"{indent}  {line}\n"

                        # 递归处理子循环
                        if 'children' in loop:
                            code += generate_loop_code(loop['children'], indent_level + 2)

                        code += f"{indent}END DO\n"
                    elif loop['type'] == 'while':
                        code += f"{indent}DO WHILE ({loop['condition']})\n"

                        # 添加循环体
                        for line in loop['body']:
                            code += f"{indent}  {line}\n"

                        code += f"{indent}END DO\n"
                    else:  # simple loop
                        code += f"{indent}DO\n"

                        # 添加循环体
                        for line in loop['body']:
                            code += f"{indent}  {line}\n"

                        code += f"{indent}END DO\n"

                return code

            self.cuda_code += generate_loop_code(remaining_loops)
        else:
            # 如果所有循环都映射到线程，直接添加循环体
            self.cuda_code += "\n      ! 原始代码的核心逻辑\n"
            for loop in mapped_loops:
                for line in loop['body']:
                    if not line.strip().lower().startswith("do ") and not line.strip().lower().startswith("end"):
                        self.cuda_code += f"      {line}\n"

        # 关闭边界检查块
        if bounds_check:
            self.cuda_code += "    END IF\n"

        # 结束内核
        self.cuda_code += "  END SUBROUTINE\nEND MODULE"

        return self.cuda_code

    def generate_host_code(self):
        """
        生成主机端代码以调用CUDA内核
        """
        # 确定线程块和网格维度
        mapped_loops = self.parallelizable_loops[:3]
        block_sizes = []
        grid_sizes = []

        for i, loop in enumerate(mapped_loops):
            if i >= 3:  # CUDA最多3D线程块
                break

            # 计算循环范围
            start = loop['start']
            end = loop['end']

            # 默认块大小
            block_size = 32 if i == 0 else 16 if i == 1 else 4

            block_sizes.append(str(block_size))

            # 计算网格大小 (向上取整到块大小的倍数)
            grid_expr = f"(({end} - {start} + 1) + {block_size} - 1) / {block_size}"
            grid_sizes.append(grid_expr)

        # 如果不足3维，用1补齐
        while len(block_sizes) < 3:
            block_sizes.append("1")
            grid_sizes.append("1")

        # 生成主机代码
        host_code = f"""SUBROUTINE run_{self.kernel_name}()
  USE cuda_module

  ! 声明变量
  ! [变量声明需要在这里添加]

  ! 定义线程块和网格维度
  TYPE(dim3) :: grid, block

  ! 设置线程块和网格维度
  block = dim3({block_sizes[0]}, {block_sizes[1]}, {block_sizes[2]})
  grid = dim3({grid_sizes[0]}, {grid_sizes[1]}, {grid_sizes[2]})

  ! 调用内核
  CALL {self.kernel_name}<<<grid, block>>>({', '.join(var for var in self.parser.variables)})

  ! 同步设备
  istat = cudaDeviceSynchronize()

END SUBROUTINE"""

        return host_code


def convert_fortran_to_cuda(source_code):
    """
    将Fortran代码转换为CUDA代码
    """
    # 解析Fortran代码
    parser = FortranParser(source_code)
    parser.lexical_analysis()
    parser.detect_declarations()
    parser.syntax_parsing()
    parser.ast_construction()
    parser.dependency_analysis()

    # 生成CUDA代码
    generator = CUDAGenerator(parser)
    cuda_kernel = generator.generate_cuda_code()
    host_code = generator.generate_host_code()

    return {
        'cuda_kernel': cuda_kernel,
        'host_code': host_code
    }


def main():

    fortran_code = """
    !
    program matrix_multiply
      implicit none
      integer, parameter :: n = 1000
      real :: a(n,n), b(n,n), c(n,n)
      integer :: i, j, k

      ! 初始化矩阵
      do i = 1, n
        do j = 1, n
          a(i,j) = i + j
          b(i,j) = i * j
          c(i,j) = 0.0
        end do
      end do

      ! 执行矩阵乘法
      do i = 1, n
        do j = 1, n
          do k = 1, n
            c(i,j) = c(i,j) + a(i,k) * b(k,j)
          end do
        end do
      end do

      print *, "Matrix multiplication completed."
    end program matrix_multiply
    """

    # 转换为CUDA
    result = convert_fortran_to_cuda(fortran_code)

    print("原始Fortran代码:")
    print("-" * 50)
    print(fortran_code)

    print("\n生成的CUDA内核代码:")
    print("-" * 50)
    print(result['cuda_kernel'])

    print("\n生成的主机端代码:")
    print("-" * 50)
    print(result['host_code'])


if __name__ == "__main__":
    main()