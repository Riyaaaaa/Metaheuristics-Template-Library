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
#include"../configuration.h"

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
        f.template operator()<index>(std::forward<Tuple>(tuple),std::forward<Args>(args)...);
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

template<size_t index, size_t end, bool isEnd = index == end>
struct _static_for;

template<size_t index, size_t end>
struct _static_for<index, end, false>
{
    template<typename F, typename... Args>
    static void Execute(F&& f, Args&&... args)
    {
        f.template operator()<index>(std::forward<Args>(args)...);
        _static_for<index+1,end>::Execute(std::forward<F>(f),std::forward<Args>(args)...);
    }
};

template<size_t index, size_t end>
struct _static_for<index, end, true>
{
    template<class F, typename... Args>
    static void Execute(F&& f, Args&&... args)
    {
    }
};

template<std::size_t begin,std::size_t end,class F,class... Args>
void static_for(F&& f,Args... args){
    _static_for<begin,end>::Execute(std::forward<F>(f),std::forward<Args>(args)...);
}

template<size_t index, size_t end,template<std::size_t...>class F,bool isEnd = index == end,std::size_t... indexes>
struct _static_for_nested;

template<size_t index, size_t end,template<std::size_t...>class F,std::size_t... indexes>
using _static_for_nested_impl = _static_for_nested<index,end,F,index==end,indexes...>;

template<size_t index, size_t end,template<std::size_t...>class F,std::size_t... indexes>
struct _static_for_nested<index, end, F, false, indexes...>
{
    template<typename... Args>
    static void Execute(Args&&... args)
    {
        F<index,indexes...> f;
        f(std::forward<Args>(args)...);
        _static_for_nested_impl<index+1, end, F, indexes...>::Execute(std::forward<Args>(args)...);
    }
};

template<size_t index, size_t end,template<std::size_t...>class F,std::size_t... indexes>
struct _static_for_nested<index, end, F, true, indexes...>
{
    template<typename... Args>
    static void Execute(Args&&... args)
    {
    }
};

template<std::size_t begin,std::size_t end,template<std::size_t...>class F,std::size_t... indexes,class... Args>
void static_for_nested(Args&&... args){
    _static_for_nested<begin, end, F, begin==end, indexes...>::Execute(std::forward<Args>(args)...);
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
template < class Ty, std::size_t size >
struct array_base < std::array < Ty, size > >
{
    using type = Ty;
};
template < class Ty >
using array_base_t = typename array_base < Ty >::type;

LIB_SCOPE_END()

#endif
