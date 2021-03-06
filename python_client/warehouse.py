# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: warehouse.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import Dict, List, Optional

import betterproto
import grpclib


@dataclass
class CreateWorkspaceRequest(betterproto.Message):
    """/ Request to create a workspace"""

    # / Name for the new workspace
    name: str = betterproto.string_field(1)
    # / Metadata associated with the workspace
    metadata: "ObjectMetadata" = betterproto.message_field(2)


@dataclass
class UpdateWorkspaceRequest(betterproto.Message):
    """/ Request to update information about a workspace"""

    # / Workspace to update
    workspace: "WorkspaceId" = betterproto.message_field(1)
    # / New metadata for the workspace
    metadata: "ObjectMetadata" = betterproto.message_field(2)


@dataclass
class DeleteObjectRequest(betterproto.Message):
    """/ Request to delete an object from a workspace"""

    # / Workspace to remove the object from
    workspace: "WorkspaceId" = betterproto.message_field(1)
    # / Path to the object to remove
    path: "WorkspacePath" = betterproto.message_field(2)


@dataclass
class CreateObjectRequest(betterproto.Message):
    """/ Request to create an object inside a workspace"""

    # / Workspace to create new object in
    workspace: "WorkspaceId" = betterproto.message_field(1)
    # / New object to add to the workspace
    object: "WorkspaceObject" = betterproto.message_field(2)


@dataclass
class WorkspacePath(betterproto.Message):
    """/ Path to an item in a workspace"""

    parts: List[str] = betterproto.string_field(1)


@dataclass
class WorkspaceId(betterproto.Message):
    """/ Wrapper type around a workspace ID (subject to change)"""

    name: str = betterproto.string_field(1)


@dataclass
class ObjectMetadata(betterproto.Message):
    """/ Basic metadata about the workspace"""

    # / Brief description of the workspace
    description: str = betterproto.string_field(2)
    # / UTC timestamp in seconds since UNIX_EPOCH
    created_at: int = betterproto.uint64_field(3)
    # / Creator of the workspace
    created_by: str = betterproto.string_field(4)
    # / Any extra information about the workspace
    additional_info: Dict[str, str] = betterproto.map_field(
        5, betterproto.TYPE_STRING, betterproto.TYPE_STRING
    )


@dataclass
class Workspace(betterproto.Message):
    """/ Top level workspace object,"""

    id: "WorkspaceId" = betterproto.message_field(1)
    # / Metadata for the workspace as a whole
    metadata: "ObjectMetadata" = betterproto.message_field(2)
    # / Objects contained under this workspace
    objects: List["WorkspaceObject"] = betterproto.message_field(3)


@dataclass
class WorkspaceObject(betterproto.Message):
    """
    / Object that lives inside a workspace. Either a nested folder or table
    """

    # / Name of the workspace object
    name: str = betterproto.string_field(1)
    # / Path to the object in the workspace
    path: "WorkspacePath" = betterproto.message_field(2)
    # / Metadata for the object
    metadata: "ObjectMetadata" = betterproto.message_field(3)
    table: "WorkspaceTable" = betterproto.message_field(4, group="data")
    collection: "WorkspaceCollection" = betterproto.message_field(5, group="data")


@dataclass
class WorkspaceTable(betterproto.Message):
    # / Arrow IPC schema representation as bytes
    schema: bytes = betterproto.bytes_field(1)


@dataclass
class WorkspaceCollection(betterproto.Message):
    # / Children under the collection
    children: List["WorkspaceObject"] = betterproto.message_field(3)


@dataclass
class Empty(betterproto.Message):
    pass


class WarehouseStub(betterproto.ServiceStub):
    """
    High level warehouse service to manage workspaces.Any data transfer should
    be done through the flight endpoints
    """

    async def create_workspace(
        self, *, name: str = "", metadata: Optional["ObjectMetadata"] = None
    ) -> Workspace:
        """/ Creates a new workspace in the warehouse"""

        request = CreateWorkspaceRequest()
        request.name = name
        if metadata is not None:
            request.metadata = metadata

        return await self._unary_unary(
            "/warehouse.Warehouse/CreateWorkspace",
            request,
            Workspace,
        )

    async def delete_workspace(self, *, name: str = "") -> Empty:
        """/ Deletes a workspace from the warehouse and all of its data"""

        request = WorkspaceId()
        request.name = name

        return await self._unary_unary(
            "/warehouse.Warehouse/DeleteWorkspace",
            request,
            Empty,
        )

    async def update_workspace(
        self,
        *,
        workspace: Optional["WorkspaceId"] = None,
        metadata: Optional["ObjectMetadata"] = None,
    ) -> Workspace:
        """/ Updates information about a workspace"""

        request = UpdateWorkspaceRequest()
        if workspace is not None:
            request.workspace = workspace
        if metadata is not None:
            request.metadata = metadata

        return await self._unary_unary(
            "/warehouse.Warehouse/UpdateWorkspace",
            request,
            Workspace,
        )

    async def get_workspace(self, *, name: str = "") -> Workspace:
        """/ Gets information about a workspace"""

        request = WorkspaceId()
        request.name = name

        return await self._unary_unary(
            "/warehouse.Warehouse/GetWorkspace",
            request,
            Workspace,
        )

    async def delete_workspace_object(
        self,
        *,
        workspace: Optional["WorkspaceId"] = None,
        path: Optional["WorkspacePath"] = None,
    ) -> Empty:
        """/ Request to delete an object from a workspace"""

        request = DeleteObjectRequest()
        if workspace is not None:
            request.workspace = workspace
        if path is not None:
            request.path = path

        return await self._unary_unary(
            "/warehouse.Warehouse/DeleteWorkspaceObject",
            request,
            Empty,
        )

    async def create_workspace_object(
        self,
        *,
        workspace: Optional["WorkspaceId"] = None,
        object: Optional["WorkspaceObject"] = None,
    ) -> Empty:
        """/ Request to create a new workspace object"""

        request = CreateObjectRequest()
        if workspace is not None:
            request.workspace = workspace
        if object is not None:
            request.object = object

        return await self._unary_unary(
            "/warehouse.Warehouse/CreateWorkspaceObject",
            request,
            Empty,
        )
