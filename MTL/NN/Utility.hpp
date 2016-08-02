//
//  Utility.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/14.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_Utility_h
#define MTL_Development_Utility_h

#include<utility>
#include<tuple>
#include<array>
#include<vector>
#include<list>
#include<map>
#include<deque>
#include<set>
#include<unordered_map>
#include<unordered_set>
#include"../Common/configuration.h"

LIB_SCOPE_BEGIN()

/* part recursive*/
template<size_t begin, size_t end, bool terminate = begin + 1 == end>
struct surfaceExecutePart;

template<size_t begin, size_t end>
struct surfaceExecutePart<begin, end, true>
{
    template<typename Tuple, typename Function, typename... Args>
    static void Execute(Tuple && tuple, Function && function, Args&&... args)
    {
        function(std::get<begin>(tuple),std::forward<Args>(args)...);
    }
};

template<size_t begin, size_t end>
struct surfaceExecutePart<begin, end, false>
{
    template<typename Tuple, typename Function, typename... Args>
    static void Execute(Tuple&& tuple, Function && function, Args&&... args)
    {
        surfaceExecutePart<begin, (begin + end) / 2>::Execute
        (std::forward<Tuple>(tuple), std::forward<Function>(function), std::forward<Args>(args)...);
        
        surfaceExecutePart<(begin + end) / 2, end>::Execute
        (std::forward<Tuple>(tuple), std::forward<Function>(function), std::forward<Args>(args)...);
    }
};

// pass all element of tuple to function
template<std::size_t begin, std::size_t end,typename Tuple, typename F, typename... Args>
void surfaceExecuteAll(Tuple&& tuple, F&& function, Args&&... args)
{
    surfaceExecutePart<begin, end>::Execute(
                                            std::forward<Tuple>(tuple),
                                            std::forward<F>(function),
                                            std::forward<Args>(args)...
                                            );
}

/* forward unfold */
template<size_t index, size_t end, bool isEnd = index == end>
struct forwardExecute;

template<size_t index, size_t end>
struct forwardExecute<index, end, false>
{
    template<typename Tuple, typename F, typename... Args>
    static void Execute(Tuple&& tuple, F&& f, Args&&... args)
    {
        f(std::get<index>(tuple),std::forward<Args>(args)...);
        forwardExecute<index+1,end>::Execute(
                                             std::forward<Tuple>(tuple),
                                             std::forward<F>(f),
                                             std::forward<Args>(args)...
                                             );
        
    }
};

template<size_t index, size_t end>
struct forwardExecute<index, end, true>
{
    template<typename Tuple, class F, typename... Args>
    static void Execute(Tuple&& tuple, F&& f, Args&&... args)
    {
    }
};

// pass all element of tuple to function
template< std::size_t begin, std::size_t end,typename Tuple, typename Function,typename... Args>
void forwardExecuteAll(Tuple&& tuple, Function&& function, Args&&... args)
{
    forwardExecute<begin, end>::Execute(
                                        std::forward<Tuple>(tuple),
                                        std::forward<Function>(function),
                                        std::forward<Args>(args)...
                                        );
}

/* recursive and propagate return value. */
template<std::size_t index>
struct propagationTuple{
    template<class Tuple, class F, class R,typename... Args>
    static auto Execute(Tuple&& tuple,
                        F&& f,
                        R&& result,
                        Args&&... args){
        return propagationTuple<index-1>::Execute(std::forward<Tuple>(tuple),
                                                  std::forward<F>(f),
                                                  std::forward<F>(f)(std::get<index>(tuple),
                                                                     std::forward<Args>(args)...,
                                                                     std::forward<R>(result)
                                                                     ),
                                                  std::forward<Args>(args)...
                                                  );
    }
};

template<>
struct propagationTuple<0>{
    template<class Tuple,class F, class R,typename... Args>
    static auto Execute(Tuple&& tuple,
                        F&&     f,
                        R&&     result,
                        Args&&... args)
    {
        return std::forward<F>(f)(std::get<0>(tuple),std::forward<Args>(args)...,std::forward<R>(result));
    }
};

template<std::size_t index,class Tuple, class F, typename... Args>
auto propagationTupleApply(Tuple&& tuple,F&& f, Args&&... args){
    return propagationTuple<index-1>::Execute(std::forward<Tuple>(tuple),
                                              std::forward<F>(f),
                                              std::forward<F>(f)(std::get<index>(tuple),std::forward<Args>(args)...),
                                              std::forward<Args>(args)...
                                              );
}

namespace detail {
    
    template<size_t index, size_t end, bool isEnd = index == end - 1>
    struct static_for;
    
    template<size_t index, size_t end>
    struct static_for<index, end, false>
    {
        template<typename F, typename... Args>
        static void Execute(F&& f, Args&&... args)
        {
            f.template operator()<index>(std::forward<Args>(args)...);
            static_for<index + 1, end>::Execute(std::forward<F>(f), std::forward<Args>(args)...);
        }
    };
    
    template<size_t index, size_t end>
    struct static_for<index, end, true>
    {
        template<class F, typename... Args>
        static void Execute(F&& f, Args&&... args)
        {
            f.template operator()<index>(std::forward<Args>(args)...);
        }
    };
    
    template<size_t index, size_t end, class F, bool isEnd, std::size_t... indexes>
    struct static_for_nested_impl;
    
    template<size_t index, size_t end,class F, std::size_t... indexes>
    using static_for_nested = static_for_nested_impl<index, end, F, index == end - 1, indexes...>;
    
    template<size_t index, size_t end, class F, std::size_t... indexes>
    struct static_for_nested_impl<index, end, F, false, indexes...>
    {
        template<typename... Args>
        static void Execute(F&& f,Args&&... args)
        {
            f.template operator()<index, indexes...>(std::forward<Args>(args)...);
            static_for_nested<index + 1, end, F, indexes...>::Execute(std::forward<F>(f),std::forward<Args>(args)...);
        }
    };
    
    template<size_t index, size_t end ,class F, std::size_t... indexes>
    struct static_for_nested_impl<index, end, F, true, indexes...>
    {
        template<typename... Args>
        static void Execute(F&& f,Args&&... args)
        {
            f.template operator()<index, indexes...>(std::forward<Args>(args)...);
        }
    };
}

template<std::size_t begin, std::size_t end, class F, class... Args>
void static_for(F&& f,Args&&... args){
    detail::static_for<begin,end>::Execute(std::forward<F>(f),std::forward<Args>(args)...);
}

template<std::size_t begin, std::size_t end, std::size_t... indexes, class F, class... Args>
void static_for_nested(F&& f, Args&&... args){
    detail::static_for_nested<begin, end, F, indexes...>::Execute(std::forward<F>(f),std::forward<Args>(args)...);
}
/* concatenate std::tuple */
template <class Seq1, class Seq2>
struct connect;

template <class... Seq1, class... Seq2>
struct connect<std::tuple<Seq1...>, std::tuple<Seq2...>> {
    typedef std::tuple<Seq1..., Seq2...> type;
};

/* make std::tuple< std::array<T,Size1>, std::array<T,Size2>, ... std::array<T,SizeN> > */
template<class T, class Tuple,std::size_t... Dims>
struct make_tuple_array{
    typedef Tuple type;
};

template<class T, class Tuple, std::size_t First, std::size_t... Dims>
struct make_tuple_array<T,Tuple,First,Dims...>{
    typedef typename make_tuple_array< T, typename mtl::connect < Tuple, std::tuple<std::array<T,First>> >::type  , Dims... >::type type;
};

/* T has template argments */
template<template<std::size_t>class T, class Tuple,std::size_t... Dims>
struct make_tuple_array_3dims{
    typedef Tuple type;
};

template<template<std::size_t>class T, class Tuple, std::size_t First, std::size_t... Dims>
struct make_tuple_array_3dims<T,Tuple,First,Dims...>{
    typedef typename make_tuple_array_3dims< T, typename mtl::connect < Tuple, std::tuple<std::array<T<1>,First>> >::type  , Dims... >::type type;
};

template<template<std::size_t>class T, class Tuple, std::size_t First, std::size_t Next, std::size_t... Dims>
struct make_tuple_array_3dims<T,Tuple,First,Next,Dims...>{
    typedef typename make_tuple_array_3dims< T, typename mtl::connect < Tuple, std::tuple<std::array<T<Next>,First> > >::type  , Next , Dims... >::type type;
};

// Get Base Type of Array
template < class >
struct array_base
{ // Ty must be array
};
template < class Ty, std::size_t size >
struct array_base < Ty[size] >
{
    using type = Ty;
};
template < template<class T, std::size_t s> class Array, class Ty, std::size_t size >
struct array_base < Array < Ty, size > >
{
    using type = Ty;
};

template < template<class T, class A> class Array, class Ty, class Allocator >
struct array_base < Array < Ty, Allocator > >
{
	using allocator = Allocator;
	using type = Ty;
};


template < class Ty >
using array_base_t = typename array_base < Ty >::type;

template <template <class...> class T, template <class...> class U>
struct is_same_template : std::false_type {};

template <template <class...> class T>
struct is_same_template<T, T> : std::true_type {};

template <template <class...> class T>
struct is_array_template {
	static constexpr bool value =
		is_same_template<T, std::vector>::value ||
		is_same_template<T, std::list>::value ||
		is_same_template<T, std::deque>::value ||
		is_same_template<T, std::set>::value ||
		is_same_template<T, std::unordered_set>::value;
};

template <template <class...> class T>
struct is_object_template {
	static constexpr bool value =
		is_same_template<T, std::map>::value ||
		is_same_template<T, std::unordered_map>::value;
};

template <template <class...> class T>
struct is_container_template {
	static constexpr bool value =
		is_array_template<T>::value || is_object_template<T>::value;
};

template <class E>
struct is_container {
	static constexpr bool value = false;
};

template <template <class...> class T, class... Args>
struct is_container<T<Args...>> : is_container_template<T> {};

//template< template<class...> class Container, class T, class Allocator>
//struct _Container_Initializer;
//
//template< template<class...> class Container, class T = typename array_base<Container>::type, class Allocator = typename array_base<Container>::allocator >
//struct Container_Initializer_traits{
//	using type = _Container_Initializer<Container, T, Allocator>;
//};
//
//template< template<class...> class Container >
//using Container_Initializer = typename Container_Initializer_traits<Container>::type;
//
//template< template<class...> class Container, class T, class Allocator>
//struct _Container_Initializer{
//	static void init(Container<T,Allocator> container, T value) {
//		std::fill(container.begin(), container.end(), value);
//	}
//};
//
//template< template<class> class Container, template<class...> class InnerContainer, class T, class Allocator >
//struct _Container_Initializer<Container, InnerContainer<T,Allocator>, Allocator>{
//	static void init( Container< InnerContainer<T, Allocator>, Allocator> container, T value) {
//		for (auto&& n : container) {
//			Container_Initializer< InnnerContainer >::init(n, value);
//		}
//	}
//};



LIB_SCOPE_END()

#endif
