"""基于字典树(路径 Trie)的内存文件系统。

来源: 腾讯-TEG-AI-Infra-一二三面(一面 编程 27)。参考 LeetCode 1166/588。

设计: 每个节点是一个路径组件(目录或文件), children dict 即 Trie 的 next 指针表;
'/' split 后逐段下行, 操作代价 O(depth * len(path))。
支持的边界语义:
- mkdir: 父目录必须存在; 中间已存在同名文件 -> 报错; recursive=True 时逐级创建;
- write/read/touch: 父必须存在且为目录, 目标不能是目录;
- rm: 目录默认须为空, recursive=True 递归删;
- mv: 目标父目录须存在, 不能移动到自身子目录(防环)。
"""
from __future__ import annotations


class _FsNode:
    __slots__ = ("children", "is_file", "content")

    def __init__(self, is_file: bool = False) -> None:
        self.children: dict[str, _FsNode] = {}
        self.is_file = is_file
        self.content: str = ""


class FsError(Exception):
    """文件系统操作错误(路径不存在/类型冲突/非法操作)。"""


def _split(path: str) -> list[str]:
    if not path.startswith("/"):
        raise FsError(f"路径必须以 / 开头: {path!r}")
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise FsError("路径不能是根目录本身的可写目标")
    return parts


class InMemoryFS:
    def __init__(self) -> None:
        self.root = _FsNode()

    # ---------- 内部工具 ----------
    def _walk(self, parts: list[str]) -> tuple[_FsNode, _FsNode]:
        """返回 (父节点, 目标节点); 目标不存在时目标为 None。"""
        node = self.root
        for name in parts[:-1]:
            if name not in node.children:
                raise FsError(f"路径不存在: /{'/'.join(parts)}")
            node = node.children[name]
            if node.is_file:
                raise FsError(f"中间组件是文件: {name}")
        parent = node
        target = parent.children.get(parts[-1])
        return parent, target

    def _require_dir(self, node: _FsNode, name: str) -> None:
        if node.is_file:
            raise FsError(f"不是目录: {name}")

    # ---------- 查询 ----------
    def exists(self, path: str) -> bool:
        node = self.root
        for name in _split(path):
            if name not in node.children:
                return False
            node = node.children[name]
        return True

    def is_dir(self, path: str) -> bool:
        return not self._node_at(path).is_file

    def _node_at(self, path: str) -> _FsNode:
        parent, target = self._walk(_split(path))
        if target is None:
            raise FsError(f"路径不存在: {path}")
        return target

    def ls(self, path: str) -> list[str]:
        """目录返回字典序子项; 文件返回 [文件名]。"""
        node = self._node_at(path)
        if node.is_file:
            return [path.rstrip("/").rsplit("/", 1)[-1]]
        return sorted(node.children)

    def read_file(self, path: str) -> str:
        node = self._node_at(path)
        if not node.is_file:
            raise FsError(f"是目录不能读: {path}")
        return node.content

    def du(self, path: str = "/") -> int:
        """子树内所有文件内容长度之和。"""
        total = 0
        stack = [self.root if path == "/" else self._node_at(path)]
        while stack:
            node = stack.pop()
            if node.is_file:
                total += len(node.content)
            else:
                stack.extend(node.children.values())
        return total

    # ---------- 变更 ----------
    def mkdir(self, path: str, recursive: bool = False) -> None:
        parts = _split(path)
        node = self.root
        for i, name in enumerate(parts):
            child = node.children.get(name)
            if child is None:
                if not recursive and i != len(parts) - 1:
                    raise FsError(f"父目录不存在(可用 recursive=True): /{'/'.join(parts[:i + 1])}")
                child = _FsNode(is_file=False)
                node.children[name] = child
            elif child.is_file:
                raise FsError(f"中间组件是文件: {name}")
            elif i == len(parts) - 1:
                raise FsError(f"目录已存在: {path}")
            node = child

    def touch(self, path: str) -> None:
        parts = _split(path)
        parent, target = self._walk(parts)
        self._require_dir(parent, "父路径")
        if target is None:
            parent.children[parts[-1]] = _FsNode(is_file=True)
        elif target.is_file:
            pass  # 幂等
        else:
            raise FsError(f"是目录不能 touch: {path}")

    def write_file(self, path: str, data: str) -> None:
        parts = _split(path)
        parent, target = self._walk(parts)
        self._require_dir(parent, "父路径")
        if target is not None and not target.is_file:
            raise FsError(f"是目录不能写: {path}")
        if target is None:
            target = _FsNode(is_file=True)
            parent.children[parts[-1]] = target
        target.content = data

    def append_file(self, path: str, data: str) -> None:
        self.write_file(path, self.read_file(path) + data)

    def rm(self, path: str, recursive: bool = False) -> None:
        parts = _split(path)
        parent, target = self._walk(parts)
        if target is None:
            raise FsError(f"路径不存在: {path}")
        if not target.is_file and target.children and not recursive:
            raise FsError(f"目录非空(可用 recursive=True): {path}")
        del parent.children[parts[-1]]

    def mv(self, src: str, dst: str) -> None:
        src_parts, dst_parts = _split(src), _split(dst)
        sparent, starget = self._walk(src_parts)
        if starget is None:
            raise FsError(f"源不存在: {src}")
        dparent, dtarget = self._walk(dst_parts)
        self._require_dir(dparent, "目标父路径")
        if dtarget is not None:
            raise FsError(f"目标已存在: {dst}")
        # 防环: 目标父目录不能在源子树内
        probe = dparent
        while probe is not None:
            if probe is starget:
                raise FsError("不能把目录移动到它自己的子目录")
            probe = self._ancestor_of(probe)
        del sparent.children[src_parts[-1]]
        dparent.children[dst_parts[-1]] = starget

    def _ancestor_of(self, node: _FsNode) -> _FsNode | None:
        """在树中找 node 的父节点(小规模内存 FS 直接 DFS, 工程上可加 parent 指针)。"""
        stack = [self.root]
        while stack:
            cur = stack.pop()
            for child in cur.children.values():
                if child is node:
                    return cur
                stack.append(child)
        return None
