#pragma once

template <typename TY>
TY* create1DArray(TY *&arr, int i);

template <typename TY>
void destroy1DArray(TY *&arr);

template <typename TY>
TY** create2DArray(TY **&arr, int i, int j);

template <typename TY>
void destroy2DArray(TY **&arr);

template <typename TY>
TY*** create3DArray(TY ***&arr, int i, int j, int k);

template <typename TY>
void destroy3DArray(TY ***&arr);