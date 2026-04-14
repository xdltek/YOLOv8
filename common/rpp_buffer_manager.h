/** @file rpp_buffer_manager.h
 *
 * @brief IO buffer manager
 * @author XDLTek Technologies
 * COPYRIGHT(c) 2020-2022 XDLTek Technologies.
 * ALL RIGHTS RESERVED
 *
 * This is Unpublished Proprietary Source Code of XDLTek Technologies
 */

#ifndef RPPRT_CORE_BUILDER_BUFFER_MANAGER_H_
#define RPPRT_CORE_BUILDER_BUFFER_MANAGER_H_

#include <new>
#include <memory>
#include <stdio.h>

#include "sampleCommon.h"

#include "Infer.h"
#include "InferRuntimeCommon.h"
#include "rpp_runtime.h"

namespace samplesCommon {
//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
    template<typename AllocFunc, typename FreeFunc>
    class GenericBuffer {
    public:
        //!
        //! \brief Construct an empty buffer.
        //!
        GenericBuffer(infer1::DataType type = infer1::DataType::kFLOAT)
                : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {
        }

        //!
        //! \brief Construct a buffer with the specified allocation size in bytes.
        //!
        GenericBuffer(size_t size, infer1::DataType type)
                : mSize(size), mCapacity(size), mType(type) {
            if (!allocFn(&mBuffer, this->nbBytes())) {
                throw std::bad_alloc();
            }
        }

        GenericBuffer(GenericBuffer &&buf)
                : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType), mBuffer(buf.mBuffer) {
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mType = infer1::DataType::kFLOAT;
            buf.mBuffer = nullptr;
        }

        GenericBuffer &operator=(GenericBuffer &&buf) {
            if (this != &buf) {
                freeFn(mBuffer);
                mSize = buf.mSize;
                mCapacity = buf.mCapacity;
                mType = buf.mType;
                mBuffer = buf.mBuffer;
                // Reset buf.
                buf.mSize = 0;
                buf.mCapacity = 0;
                buf.mBuffer = nullptr;
            }
            return *this;
        }

        //!
        //! \brief Returns pointer to underlying array.
        //!
        void *data() {
            return mBuffer;
        }

        //!
        //! \brief Returns pointer to underlying array.
        //!
        const void *data() const {
            return mBuffer;
        }

        //!
        //! \brief Returns the size (in number of elements) of the buffer.
        //!
        size_t size() const {
            return mSize;
        }

        //!
        //! \brief Returns the size (in bytes) of the buffer.
        //!
        size_t nbBytes() const {
            return this->size() * samplesCommon::getElementSize(mType);
        }

        //!
        //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
        //!
        void resize(size_t newSize) {
            mSize = newSize;
            if (mCapacity < newSize) {
                freeFn(mBuffer);
                if (!allocFn(&mBuffer, this->nbBytes())) {
                    throw std::bad_alloc{};
                }
                mCapacity = newSize;
            }
        }

        //!
        //! \brief Overload of resize that accepts Dims
        //!
        void resize(const infer1::Dims &dims) {
            return this->resize(samplesCommon::volume(dims));
        }

        ~GenericBuffer() {
            freeFn(mBuffer);
        }

    private:
        size_t mSize{0}, mCapacity{0};
        infer1::DataType mType;
        void *mBuffer;
        AllocFunc allocFn;
        FreeFunc freeFn;
    };

    class DeviceAllocator {
    public:
        //!
        //! \brief Returns pointer to device memory, type: unsigned long long (RPPdeviceptr).
        //!
        bool operator()(void **ptr, size_t size) const {
            if (rtMalloc(ptr, size) != rtError_t::rtSuccess) {
                return false;
            }
            // fill with 0xff
            const auto host_buffer_ptr = std::unique_ptr<void, decltype(free)*>{ malloc(size), free };
            memset(host_buffer_ptr.get(), 0xff, size);
            rtMemcpy(*ptr, host_buffer_ptr.get(), size, rtMemcpyHostToDevice);

            return true;
        }
    };

    class DeviceFree {
    public:
        void operator()(void *ptr) const {
            if (ptr != nullptr)
            {
                checkRTError(rtFree(ptr));
            }
        }
    };


    class HostAllocator {
    public:
        bool operator()(void **ptr, size_t size) const {
            if (rtHostAlloc(ptr, size, 0) != rtError_t::rtSuccess)
            {
                return false;
            }
            // fill with 0xff
            memset(*ptr, 0xff, size);
            return true;
        }
    };

    class HostFree {
    public:
        void operator()(void *ptr) const {
            if (ptr != nullptr)
            {
                checkRTError(rtFreeHost(ptr));
            }
        }
    };

    using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
    using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
    class ManagedBuffer {
    public:
        DeviceBuffer deviceBuffer;
        HostBuffer hostBuffer;
    };

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
    class RppBufferManager {
    public:
        static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

        //!
        //! \brief Create a BufferManager for handling buffer interactions with engine.
        //!
        RppBufferManager(std::shared_ptr<infer1::IEngine> iengine, const int batchSize = 1)
                : engine_(iengine), batch_size_(batchSize) {
            // Create host and device buffers
            for (int i = 0; i < engine_->getNbBindings(); i++) {
                auto dims = engine_->getBindingDimensions(i);
                size_t vol = batchSize;
                infer1::DataType type = engine_->getBindingDataType(i);

                vol *= samplesCommon::volume(dims);
                std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};

                manBuf->deviceBuffer = DeviceBuffer(vol, type);
                manBuf->hostBuffer = HostBuffer(vol, type);

                device_bindings_.emplace_back(manBuf->deviceBuffer.data());
                managed_buffers_.emplace_back(std::move(manBuf));
            }
        }


        //!
        //! \brief Returns a vector of device buffers that you can use directly as
        //!        bindings for the execute and enqueue methods of IExecutionContext.
        //!
        std::vector<void *> &getDeviceBindings() {
            return device_bindings_;
        }

        //!
        //! \brief Returns a vector of device buffers.
        //!
        const std::vector<void *> &getDeviceBindings() const {
            return device_bindings_;
        }

        //!
        //! \brief Returns the device buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        //!
        void *getDeviceBuffer(const std::string &tensorName) const {
            return getBuffer(false, tensorName);
        }

        //!
        //! \brief Returns the host buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        //!
        void *getHostBuffer(const std::string &tensorName) const {
            return getBuffer(true, tensorName);
        }

        //!
        //! \brief Returns the size of the host and device buffers that correspond to tensorName.
        //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
        //!
        size_t size(const std::string &tensorName) const {
            int index = engine_->getBindingIndex(tensorName.c_str());
            if (index == -1)
                return kINVALID_SIZE_VALUE;
            return managed_buffers_[index]->hostBuffer.nbBytes();
        }

        void saveDataToFile(float *data, int size, std::string output_names) {
            FILE *fp_ort;
            fp_ort = fopen(output_names.c_str(), "w");

            for (auto i = 0; i < size; i++) {
                fprintf(fp_ort, "%f\n", data[i]);
            }

            fclose(fp_ort);
        }

        //!
        //! \brief Dump host buffer with specified tensorName to ostream.
        //!        Prints error message to std::ostream if no such tensor can be found.
        //!
        void dumpBuffer(std::ostream &os, size_t index) {
            void *buf = managed_buffers_[index]->hostBuffer.data();
            size_t bufSize = managed_buffers_[index]->hostBuffer.nbBytes();
            infer1::Dims bufDims = engine_->getBindingDimensions(index);
            size_t rowCount = static_cast<size_t>(bufDims.nbDims > 0 ? bufDims.d[bufDims.nbDims - 1] : batch_size_);
            int leadDim = batch_size_;
            int *trailDims = bufDims.d;
            int nbDims = bufDims.nbDims;

            // Fix explicit Dimension networks
            if (!leadDim && nbDims > 0) {
                leadDim = bufDims.d[0];
                ++trailDims;
                --nbDims;
            }

            os << "[" << std::dec << leadDim;
            for (int i = 0; i < nbDims; i++)
                os << ", " << std::dec << trailDims[i];
            os << "]" << std::endl;
            switch (engine_->getBindingDataType(index)) {
                case infer1::DataType::kINT32:
                    print<int32_t>(os, buf, bufSize, rowCount);
                    break;
                case infer1::DataType::kFLOAT:
                    print<float>(os, buf, bufSize, rowCount);
                    break;
                    // TODO: Remove it temporary
                    //case infer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break;
                case infer1::DataType::kINT8:
                    throw std::runtime_error( "Int8 network-level input and output is not supported");
//                    break;
                    // TODO: Remove it temporary
                    //case infer1::DataType::kBOOL: assert(0 && "Bool network-level input and output are not supported"); break;
                default:
                    break;
            }
        }

        //!
        //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
        //!        rowCount parameter controls how many elements are on each line.
        //!        A rowCount of 1 means that there is only 1 element on each line.
        //!
        template<typename T>
        void print(std::ostream &os, void *buf, size_t bufSize, size_t rowCount) {
            if (rowCount == 0) {
                throw std::runtime_error("row count cannot be zero.");
            }
            if (bufSize % sizeof(T) != 0) {
                throw std::runtime_error("invalid buffer size.");
            }

//            assert(rowCount != 0);
//            assert(bufSize % sizeof(T) == 0);
            T *typedBuf = static_cast<T *>(buf);

            size_t numItems = bufSize / sizeof(T);
            for (int i = 0; i < static_cast<int>(numItems); i++) {
                // Handle rowCount == 1 case
                if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                    os << typedBuf[i] << std::endl;
                else if (rowCount == 1)
                    os << typedBuf[i];
                    // Handle rowCount > 1 case
                else if (i % rowCount == 0)
                    os << typedBuf[i];
                else if (i % rowCount == rowCount - 1)
                    os << " " << typedBuf[i] << std::endl;
                else
                    os << " " << typedBuf[i];
            }
        }

        //!
        //! \brief Copy the contents of input host buffers to input device buffers synchronously.
        //!
        void copyInputToDevice() {
            memcpyBuffers(true, false, false);
        }

        //!
        //! \brief Copy the contents of output device buffers to output host buffers synchronously.
        //!
        void copyOutputToHost() {
            memcpyBuffers(false, true, false);
        }

        //!
        //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
        //!
        void copyInputToDeviceAsync(const cudaStream_t &stream = 0) {
            memcpyBuffers(true, false, true, stream, false);
        }

        //!
        //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
        //!
        void copyOutputToHostAsync(const cudaStream_t &stream = 0) {
            memcpyBuffers(false, true, true, stream, false);
        }

        //!
        //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
        //!
        void copyInputToDeviceAsync_EB(const cudaStream_t &stream = 0) {
            memcpyBuffers(true, false, true, stream, true);
        }

        ~RppBufferManager() = default;

    private:
        void *getBuffer(const bool isHost, const std::string &tensorName) const {
            int index = engine_->getBindingIndex(tensorName.c_str());
            if (index == -1)
                return nullptr;
            return (isHost ? managed_buffers_[index]->hostBuffer.data() : managed_buffers_[index]->deviceBuffer.data());
        }

        // TODO： EB only
        void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const rtStream_t &stream = 0, bool force_bf16 = false) {
            for (int i = 0; i < engine_->getNbBindings(); i++) {
                void *dstPtr
                        = deviceToHost ? managed_buffers_[i]->hostBuffer.data()
                                       : managed_buffers_[i]->deviceBuffer.data();
                const void *srcPtr
                        = deviceToHost ? managed_buffers_[i]->deviceBuffer.data()
                                       : managed_buffers_[i]->hostBuffer.data();
                size_t byteSize = managed_buffers_[i]->hostBuffer.nbBytes();

                if (force_bf16)
                {
                    byteSize = byteSize / 2;
                }

                const rtMemcpyKind memcpyType = deviceToHost ? rtMemcpyKind::rtMemcpyDeviceToHost
                                                             : rtMemcpyKind::rtMemcpyHostToDevice;

                if ((copyInput && engine_->bindingIsInput(i)) || (!copyInput && !engine_->bindingIsInput(i))) {
                    if (async) {
                        checkRTError(rtMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                    } else {
                        checkRTError(rtMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
                    }
                }
            }
        }

        std::shared_ptr<infer1::IEngine> engine_;                  //!< The pointer to the engine
        int batch_size_;                                              //!< The batch size for legacy networks, 0 otherwise.
        std::vector<std::unique_ptr<ManagedBuffer>> managed_buffers_; //!< The vector of pointers to managed buffers
        std::vector<void *> device_bindings_;                          //!< The vector of device buffers needed for engine execution
    };
}
#endif // RPPRT_CORE_BUILDER_BUFFER_MANAGER_H_
