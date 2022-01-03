from .warehouse import *
from grpclib.client import Channel
from typing import Dict
from datetime import datetime

import asyncio
import pyarrow as pa
import pyarrow.lib as palib
import pyarrow.flight as flight
import pandas as pd
import io


class WarehouseClient(object):
    def __init__(self, server: str = "localhost", port=5201) -> None:
        super().__init__()
        self._server = server
        self._flight_client = flight.FlightClient("grpc://{}:{}".format(server, port))
        self._warehouse_client = WarehouseStub(Channel(server, port=port))

    async def create_workspace(
        self, name: str, description: str = "", info: Dict[str, str] = {}
    ) -> Workspace:
        ws  = await self._warehouse_client.create_workspace(
            name=name,
            metadata=ObjectMetadata(
                description=description,
                created_by="USER",
                additional_info=info,
                created_at=int(datetime.now().timestamp()),
            ),
        )
        return ws

    async def show_workspace(self, name: str):
        workspace = await self.get_workspace(name)
        print('=' * 25, 'Workspace:', workspace.id.name, '=' * 25)
        for object in workspace.objects:
            print()
            self._print_workspace_object(object)

    def _print_workspace_object(self, obj: WorkspaceObject, depth=0):
        obj_type, _ = betterproto.which_one_of(obj, 'data')   
        print('\t'*depth, 'Name:', obj.name)
        print('\t'*depth, 'Path:', '"{}"'.format('/'.join(obj.path.parts)))
        print('\t'*depth, 'Type:', obj_type)     
        if obj_type == 'collection':
            for child in obj.collection.children:
                print()
                self._print_workspace_object(child, depth+1)
            

    async def delete_workspace(self, name: str):
        await self._warehouse_client.delete_workspace(name=name)

    async def create_table(
        self, workspace: str, name: str, table: pd.DataFrame, description: str = ""
    ):
        '''
            Creates a new table in the provided workspace
            workspace: str
                The name of the workspace to create the table under
            name: str
                The name (or unix style path) to create the table under
                Example:
                    my_table
                    group_a/my_table or group_a/group_b/my_table
        '''
        pa_table = pa.Table.from_pandas(table)
        # Upload the table using flight
        path_parts = name.split('/')
        flight_path = [workspace] + path_parts
        writer, _ = self._flight_client.do_put(flight.FlightDescriptor.for_path(*flight_path), pa_table.schema)
        writer.write_table(pa_table)
        writer.close()

    async def delete_object(self, workspace: str, name: str):
        await self._warehouse_client.delete_workspace_object(
            workspace=WorkspaceId(name=workspace),
            path=WorkspacePath(parts=name.split('/'))
        )

    async def get_workspace(self, workspace: str) -> Workspace:
        return await self._warehouse_client.get_workspace(name=workspace)

    def get_table(self, workspace: str, name: str) -> pd.DataFrame:
        # Fetch all the flight endpoints 
        path_parts = name.split('/')
        flight_path = [workspace] + path_parts
        descriptor = flight.FlightDescriptor.for_path(*flight_path)
        flight_info = self._flight_client.get_flight_info(descriptor)
        endpoints = flight_info.endpoints
        dfs = []
        for ep in endpoints:
            df = self._flight_client.do_get(ep.ticket).read_pandas()
            dfs.append(df)
        return pd.concat(dfs, axis=0)


