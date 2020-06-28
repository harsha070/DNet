import logging
from easydict import EasyDict
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm


class FlowProperty(Enum):
    CONTINUITY = 1
    X_MOMENTUM = 2
    Y_MOMENTUM = 3


def _loss(flow, y_true, y_pred):
    # compute continuity residual
    loss = compute_residual(flow, FlowProperty.CONTINUITY, 0.0, y_pred)
    # compute x momentum residual
    loss += compute_residual(flow, FlowProperty.X_MOMENTUM, 1.0 / flow.reynolds_number, y_pred)
    # compute y momentum residual
    loss += compute_residual(flow, FlowProperty.Y_MOMENTUM, 1.0 / flow.reynolds_number, y_pred)
    
    return loss


def compute_residual(flow, property_type, diffusion_coefficient, y_pred):
    logging.debug("y_pred {}".format(y_pred.shape))
    u_pred_batch, v_pred_batch, p_pred_batch = y_pred[:, 0, :, :], y_pred[:, 1, :, :], y_pred[:, 2, :, :]
    
    logging.debug("u_pred {} v_pred {} p_pred {}".format(u_pred_batch.shape, v_pred_batch.shape, p_pred_batch.shape))
    
    u_pred, v_pred, p_pred = u_pred_batch[0], v_pred_batch[0], p_pred_batch[0]
    
    residual_pred = 0.0
    
    for i in range(flow.n):
        for j in range(flow.n):
            u_east, v_east = vel_east(flow, u_pred, v_pred, i, j)
            u_west, v_west = vel_west(flow, u_pred, v_pred, i, j)
            u_north, v_north = vel_north(flow, u_pred, v_pred, i, j)
            u_south, v_south = vel_south(flow, u_pred, v_pred, i, j)
            
            if property_type == FlowProperty.CONTINUITY:
                residual_pred += (u_east - u_west + v_north - v_south)
                residual_pred -= diffusion_coefficient * 0.0
            
            if property_type == FlowProperty.X_MOMENTUM:
                residual_pred += (u_east ** 2 - u_west ** 2 + v_north * u_north - v_south * u_south)
                residual_pred -= diffusion_coefficient * gradient_east(flow, property_type, u_pred, i, j)
                residual_pred -= diffusion_coefficient * gradient_north(flow, property_type, u_pred, i, j)
                residual_pred += diffusion_coefficient * gradient_west(flow, property_type, u_pred, i, j)
                residual_pred += diffusion_coefficient * gradient_south(flow, property_type, u_pred, i, j)
                
                residual_pred -= (grad_pressure_x(flow, p_pred, i, j) * flow.delta_x)
            
            if property_type == FlowProperty.Y_MOMENTUM:
                residual_pred += (u_east * v_east - u_west * v_west + v_north ** 2 - v_south ** 2)
                residual_pred -= diffusion_coefficient * gradient_east(flow, property_type, v_pred, i, j)
                residual_pred -= diffusion_coefficient * gradient_north(flow, property_type, v_pred, i, j)
                residual_pred += diffusion_coefficient * gradient_west(flow, property_type, v_pred, i, j)
                residual_pred += diffusion_coefficient * gradient_south(flow, property_type, v_pred, i, j)
                
                residual_pred -= (grad_pressure_y(flow, p_pred, i, j) * flow.delta_x)
                residual_pred += (flow.gravity * flow.density)
    
    return residual_pred ** 2


def vel_south(flow, u_pred, v_pred, i, j):
    if j > 0:
        return (u_pred[i][j] + u_pred[i][j - 1]) / 2, (v_pred[i][j] + v_pred[i][j - 1]) / 2
    return 0.0, 0.0


def vel_north(flow, u_pred, v_pred, i, j):
    if j < flow.n - 1:
        return (u_pred[i][j] + u_pred[i][j + 1]) / 2, (v_pred[i][j] + v_pred[i][j + 1]) / 2
    return flow.lid_velocity, 0.0


def vel_east(flow, u_pred, v_pred, i, j):
    if i < flow.n - 1:
        return (u_pred[i][j] + u_pred[i + 1][j]) / 2, (v_pred[i][j] + v_pred[i + 1][j]) / 2
    return 0.0, 0.0


def vel_west(flow, u_pred, v_pred, i, j):
    if i > 0:
        return (u_pred[i][j] + u_pred[i][j - 1]) / 2, (v_pred[i][j] + v_pred[i][j - 1]) / 2
    return 0.0, 0.0


def grad_pressure_x(flow, p_pred, i, j):
    if i == 0 and j == 0:
        return p_pred[i][j] / (2.0 * flow.delta_x)
    if 0 < i < flow.n - 1:
        return (p_pred[i+1][j] - p_pred[i-1][j]) / (2.0 * flow.delta_x)
    if i == flow.n - 1:
        return (p_pred[i][j] - p_pred[i-1][j]) / flow.delta_x
    if i == 0:
        return (p_pred[i+1][j] - p_pred[i][j]) / flow.delta_x


def grad_pressure_y(flow, p_pred, i, j):
    if i == 0 and j == 0:
        return p_pred[i][j] / (2.0 * flow.delta_y)
    if 0 < j < flow.n - 1:
        return (p_pred[i][j+1] - p_pred[i][j-1]) / (2.0 * flow.delta_y)
    if j == flow.n - 1:
        return (p_pred[i][j] - p_pred[i][j-1]) / flow.delta_y
    if j == 0:
        return (p_pred[i][j+1] - p_pred[i][j]) / flow.delta_y


def gradient_south(flow, property_type, property_pred, i, j):
    if j > 0:
        return (property_pred[i][j] - property_pred[i][j - 1]) / flow.delta_y
    return 2 * (property_pred[i][j]) / flow.delta_y


def gradient_north(flow, property_type, property_pred, i, j):
    if j < flow.n - 1:
        return (property_pred[i][j + 1] - property_pred[i][j]) / flow.delta_y
    if property_type == FlowProperty.X_MOMENTUM:
        return 2 * (flow.lid_velocity - property_pred[i][j]) / flow.delta_y
    if property_type == FlowProperty.Y_MOMENTUM:
        return 2.0 * (0.0 - property_pred[i][j]) / flow.delta_y


def gradient_east(flow, property_type, property_pred, i, j):
    if i < flow.n - 1:
        return (property_pred[i + 1][j] - property_pred[i][j]) / 2
    return 2 * (0.0 - property_pred[i][j]) / flow.delta_x


def gradient_west(flow, property_type, property_pred, i, j):
    if i > 0:
        return (property_pred[i][j] - property_pred[i - 1][j]) / 2
    return 2.0 * (property_pred[i][j] - 0.0) / flow.delta_x
