/*
//@HEADER
// ************************************************************************
// 
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <sstream>
#include <iostream>

/*--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------*/

namespace Test {

template< class T , class ... P >
size_t allocation_count( const Kokkos::Experimental::DynRankView<T,P...> & view )
{
  const size_t card  = view.size();
  const size_t alloc = view.span();

  return card <= alloc ? alloc : 0 ;
}

/*--------------------------------------------------------------------------*/

template< typename T, class DeviceType>
struct TestViewOperator
{
  typedef DeviceType  execution_space ;

  static const unsigned N = 100 ;
  static const unsigned D = 3 ;

  typedef Kokkos::Experimental::DynRankView< T , execution_space > view_type ;

  const view_type v1 ;
  const view_type v2 ;

  TestViewOperator()
    : v1( "v1" , N , D )
    , v2( "v2" , N , D )
    {}

  static void testit()
  {
    Kokkos::parallel_for( N , TestViewOperator() );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const unsigned i ) const
  {
    const unsigned X = 0 ;
    const unsigned Y = 1 ;
    const unsigned Z = 2 ;

    v2(i,X) = v1(i,X);
    v2(i,Y) = v1(i,Y);
    v2(i,Z) = v1(i,Z);
  }
};

/*--------------------------------------------------------------------------*/

template< class DataType ,
          class DeviceType ,
          unsigned Rank >
struct TestViewOperator_LeftAndRight ;

template< class DataType , class DeviceType >
struct TestViewOperator_LeftAndRight< DataType , DeviceType , 8 >
{
  typedef DeviceType                          execution_space ;
  typedef typename execution_space::memory_space  memory_space ;
  typedef typename execution_space::size_type     size_type ;

  typedef int value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    const volatile value_type & input )
    { update |= input ; }

  KOKKOS_INLINE_FUNCTION
  static void init( value_type & update )
    { update = 0 ; }


  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutLeft, execution_space > left_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutRight, execution_space > right_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutStride, execution_space > stride_view ;

  left_view    left ;
  right_view   right ;
  stride_view  left_stride ;
  stride_view  right_stride ;
  long         left_alloc ;
  long         right_alloc ;

  TestViewOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5, unsigned N6, unsigned N7)
    : left(  "left", N0, N1, N2, N3, N4, N5, N6, N7 )
    , right( "right" , N0, N1, N2, N3, N4, N5, N6, N7 )
    , left_stride( left )
    , right_stride( right )
    , left_alloc( allocation_count( left ) )
    , right_alloc( allocation_count( right ) )
    {}

  static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5, unsigned N6, unsigned N7)
  {
    TestViewOperator_LeftAndRight driver (N0, N1, N2, N3, N4, N5, N6, N7);

    int error_flag = 0 ;

    Kokkos::parallel_reduce( 1 , driver , error_flag );

    ASSERT_EQ( error_flag , 0 );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type , value_type & update ) const
  {
    long offset ;

    offset = -1 ;
    for ( unsigned i7 = 0 ; i7 < unsigned(left.dimension_7()) ; ++i7 )
    for ( unsigned i6 = 0 ; i6 < unsigned(left.dimension_6()) ; ++i6 )
    for ( unsigned i5 = 0 ; i5 < unsigned(left.dimension_5()) ; ++i5 )
    for ( unsigned i4 = 0 ; i4 < unsigned(left.dimension_4()) ; ++i4 )
    for ( unsigned i3 = 0 ; i3 < unsigned(left.dimension_3()) ; ++i3 )
    for ( unsigned i2 = 0 ; i2 < unsigned(left.dimension_2()) ; ++i2 )
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    {
      const long j = & left( i0, i1, i2, i3, i4, i5, i6, i7 ) -
                     & left(  0,  0,  0,  0,  0,  0,  0,  0 );
      if ( j <= offset || left_alloc <= j ) { update |= 1 ; }
      offset = j ;

      if ( & left(i0,i1,i2,i3,i4,i5,i6,i7) !=
           & left_stride(i0,i1,i2,i3,i4,i5,i6,i7) ) {
        update |= 4 ;
      }
    }

    offset = -1 ;
    for ( unsigned i0 = 0 ; i0 < unsigned(right.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(right.dimension_1()) ; ++i1 )
    for ( unsigned i2 = 0 ; i2 < unsigned(right.dimension_2()) ; ++i2 )
    for ( unsigned i3 = 0 ; i3 < unsigned(right.dimension_3()) ; ++i3 )
    for ( unsigned i4 = 0 ; i4 < unsigned(right.dimension_4()) ; ++i4 )
    for ( unsigned i5 = 0 ; i5 < unsigned(right.dimension_5()) ; ++i5 )
    for ( unsigned i6 = 0 ; i6 < unsigned(right.dimension_6()) ; ++i6 )
    for ( unsigned i7 = 0 ; i7 < unsigned(right.dimension_7()) ; ++i7 )
    {
      const long j = & right( i0, i1, i2, i3, i4, i5, i6, i7 ) -
                     & right(  0,  0,  0,  0,  0,  0,  0,  0 );
      if ( j <= offset || right_alloc <= j ) { update |= 2 ; }
      offset = j ;

      if ( & right(i0,i1,i2,i3,i4,i5,i6,i7) !=
           & right_stride(i0,i1,i2,i3,i4,i5,i6,i7) ) {
        update |= 8 ;
      }
    }
  }
};

template< class DataType , class DeviceType >
struct TestViewOperator_LeftAndRight< DataType , DeviceType , 7 >
{
  typedef DeviceType                          execution_space ;
  typedef typename execution_space::memory_space  memory_space ;
  typedef typename execution_space::size_type     size_type ;

  typedef int value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    const volatile value_type & input )
    { update |= input ; }

  KOKKOS_INLINE_FUNCTION
  static void init( value_type & update )
    { update = 0 ; }


  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutLeft, execution_space > left_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutRight, execution_space > right_view ;

  left_view    left ;
  right_view   right ;
  long         left_alloc ;
  long         right_alloc ;

  TestViewOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5, unsigned N6 )
    : left(  "left" , N0, N1, N2, N3, N4, N5, N6 )
    , right( "right" , N0, N1, N2, N3, N4, N5, N6 )
    , left_alloc( allocation_count( left ) )
    , right_alloc( allocation_count( right ) )
    {}

  static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5, unsigned N6 )
  {
    TestViewOperator_LeftAndRight driver(N0, N1, N2, N3, N4, N5, N6 );

    int error_flag = 0 ;

    Kokkos::parallel_reduce( 1 , driver , error_flag );

    ASSERT_EQ( error_flag , 0 );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type , value_type & update ) const
  {
    long offset ;

    offset = -1 ;
    for ( unsigned i6 = 0 ; i6 < unsigned(left.dimension_6()) ; ++i6 )
    for ( unsigned i5 = 0 ; i5 < unsigned(left.dimension_5()) ; ++i5 )
    for ( unsigned i4 = 0 ; i4 < unsigned(left.dimension_4()) ; ++i4 )
    for ( unsigned i3 = 0 ; i3 < unsigned(left.dimension_3()) ; ++i3 )
    for ( unsigned i2 = 0 ; i2 < unsigned(left.dimension_2()) ; ++i2 )
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    {
      const long j = & left( i0, i1, i2, i3, i4, i5, i6 ) -
                     & left(  0,  0,  0,  0,  0,  0,  0 );
      if ( j <= offset || left_alloc <= j ) { update |= 1 ; }
      offset = j ;
    }

    offset = -1 ;
    for ( unsigned i0 = 0 ; i0 < unsigned(right.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(right.dimension_1()) ; ++i1 )
    for ( unsigned i2 = 0 ; i2 < unsigned(right.dimension_2()) ; ++i2 )
    for ( unsigned i3 = 0 ; i3 < unsigned(right.dimension_3()) ; ++i3 )
    for ( unsigned i4 = 0 ; i4 < unsigned(right.dimension_4()) ; ++i4 )
    for ( unsigned i5 = 0 ; i5 < unsigned(right.dimension_5()) ; ++i5 )
    for ( unsigned i6 = 0 ; i6 < unsigned(right.dimension_6()) ; ++i6 )
    {
      const long j = & right( i0, i1, i2, i3, i4, i5, i6 ) -
                     & right(  0,  0,  0,  0,  0,  0,  0 );
      if ( j <= offset || right_alloc <= j ) { update |= 2 ; }
      offset = j ;
    }
  }
};

template< class DataType , class DeviceType >
struct TestViewOperator_LeftAndRight< DataType , DeviceType , 6 >
{
  typedef DeviceType                          execution_space ;
  typedef typename execution_space::memory_space  memory_space ;
  typedef typename execution_space::size_type     size_type ;

  typedef int value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    const volatile value_type & input )
    { update |= input ; }

  KOKKOS_INLINE_FUNCTION
  static void init( value_type & update )
    { update = 0 ; }


  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutLeft, execution_space > left_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutRight, execution_space > right_view ;

  left_view    left ;
  right_view   right ;
  long         left_alloc ;
  long         right_alloc ;

  TestViewOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5 )
    : left(  "left" , N0, N1, N2, N3, N4, N5 )
    , right( "right" , N0, N1, N2, N3, N4, N5 )
    , left_alloc( allocation_count( left ) )
    , right_alloc( allocation_count( right ) )
    {}

  static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4, unsigned N5)
  {
    TestViewOperator_LeftAndRight driver (N0, N1, N2, N3, N4, N5);

    int error_flag = 0 ;

    Kokkos::parallel_reduce( 1 , driver , error_flag );

    ASSERT_EQ( error_flag , 0 );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type , value_type & update ) const
  {
    long offset ;

    offset = -1 ;
    for ( unsigned i5 = 0 ; i5 < unsigned(left.dimension_5()) ; ++i5 )
    for ( unsigned i4 = 0 ; i4 < unsigned(left.dimension_4()) ; ++i4 )
    for ( unsigned i3 = 0 ; i3 < unsigned(left.dimension_3()) ; ++i3 )
    for ( unsigned i2 = 0 ; i2 < unsigned(left.dimension_2()) ; ++i2 )
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    {
      const long j = & left( i0, i1, i2, i3, i4, i5 ) -
                     & left(  0,  0,  0,  0,  0,  0 );
      if ( j <= offset || left_alloc <= j ) { update |= 1 ; }
      offset = j ;
    }

    offset = -1 ;
    for ( unsigned i0 = 0 ; i0 < unsigned(right.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(right.dimension_1()) ; ++i1 )
    for ( unsigned i2 = 0 ; i2 < unsigned(right.dimension_2()) ; ++i2 )
    for ( unsigned i3 = 0 ; i3 < unsigned(right.dimension_3()) ; ++i3 )
    for ( unsigned i4 = 0 ; i4 < unsigned(right.dimension_4()) ; ++i4 )
    for ( unsigned i5 = 0 ; i5 < unsigned(right.dimension_5()) ; ++i5 )
    {
      const long j = & right( i0, i1, i2, i3, i4, i5 ) -
                     & right(  0,  0,  0,  0,  0,  0 );
      if ( j <= offset || right_alloc <= j ) { update |= 2 ; }
      offset = j ;
    }
  }
};

template< class DataType , class DeviceType >
struct TestViewOperator_LeftAndRight< DataType , DeviceType , 5 >
{
  typedef DeviceType                          execution_space ;
  typedef typename execution_space::memory_space  memory_space ;
  typedef typename execution_space::size_type     size_type ;

  typedef int value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    const volatile value_type & input )
    { update |= input ; }

  KOKKOS_INLINE_FUNCTION
  static void init( value_type & update )
    { update = 0 ; }


  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutLeft, execution_space > left_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutRight, execution_space > right_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutStride, execution_space > stride_view ;

  left_view    left ;
  right_view   right ;
  stride_view  left_stride ;
  stride_view  right_stride ;
  long         left_alloc ;
  long         right_alloc ;

  TestViewOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4 )
    : left(  "left" , N0, N1, N2, N3, N4 )
    , right( "right" , N0, N1, N2, N3, N4 )
    , left_stride( left )
    , right_stride( right )
    , left_alloc( allocation_count( left ) )
    , right_alloc( allocation_count( right ) )
    {}

  static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4)
  {
    TestViewOperator_LeftAndRight driver(N0, N1, N2, N3, N4);

    int error_flag = 0 ;

    Kokkos::parallel_reduce( 1 , driver , error_flag );

    ASSERT_EQ( error_flag , 0 );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type , value_type & update ) const
  {
    long offset ;

    offset = -1 ;
    for ( unsigned i4 = 0 ; i4 < unsigned(left.dimension_4()) ; ++i4 )
    for ( unsigned i3 = 0 ; i3 < unsigned(left.dimension_3()) ; ++i3 )
    for ( unsigned i2 = 0 ; i2 < unsigned(left.dimension_2()) ; ++i2 )
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    {
      const long j = & left( i0, i1, i2, i3, i4 ) -
                     & left(  0,  0,  0,  0,  0 );
      if ( j <= offset || left_alloc <= j ) { update |= 1 ; }
      offset = j ;

      if ( & left( i0, i1, i2, i3, i4 ) !=
           & left_stride( i0, i1, i2, i3, i4 ) ) { update |= 4 ; }
    }

    offset = -1 ;
    for ( unsigned i0 = 0 ; i0 < unsigned(right.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(right.dimension_1()) ; ++i1 )
    for ( unsigned i2 = 0 ; i2 < unsigned(right.dimension_2()) ; ++i2 )
    for ( unsigned i3 = 0 ; i3 < unsigned(right.dimension_3()) ; ++i3 )
    for ( unsigned i4 = 0 ; i4 < unsigned(right.dimension_4()) ; ++i4 )
    {
      const long j = & right( i0, i1, i2, i3, i4 ) -
                     & right(  0,  0,  0,  0,  0 );
      if ( j <= offset || right_alloc <= j ) { update |= 2 ; }
      offset = j ;

      if ( & right( i0, i1, i2, i3, i4 ) !=
           & right_stride( i0, i1, i2, i3, i4 ) ) { update |= 8 ; }
    }
  }
};

template< class DataType , class DeviceType >
struct TestViewOperator_LeftAndRight< DataType , DeviceType , 4 >
{
  typedef DeviceType                          execution_space ;
  typedef typename execution_space::memory_space  memory_space ;
  typedef typename execution_space::size_type     size_type ;

  typedef int value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    const volatile value_type & input )
    { update |= input ; }

  KOKKOS_INLINE_FUNCTION
  static void init( value_type & update )
    { update = 0 ; }


  typedef Kokkos::
   Experimental::DynRankView< DataType, Kokkos::LayoutLeft, execution_space > left_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutRight, execution_space > right_view ;

  left_view    left ;
  right_view   right ;
  long         left_alloc ;
  long         right_alloc ;

  TestViewOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2, unsigned N3)
    : left(  "left" , N0, N1, N2, N3 )
    , right( "right" , N0, N1, N2, N3 )
    , left_alloc( allocation_count( left ) )
    , right_alloc( allocation_count( right ) )
    {}

  static void testit(unsigned N0, unsigned N1, unsigned N2, unsigned N3)
  {
    TestViewOperator_LeftAndRight driver (N0, N1, N2, N3);

    int error_flag = 0 ;

    Kokkos::parallel_reduce( 1 , driver , error_flag );

    ASSERT_EQ( error_flag , 0 );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type , value_type & update ) const
  {
    long offset ;

    offset = -1 ;
    for ( unsigned i3 = 0 ; i3 < unsigned(left.dimension_3()) ; ++i3 )
    for ( unsigned i2 = 0 ; i2 < unsigned(left.dimension_2()) ; ++i2 )
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    {
      const long j = & left( i0, i1, i2, i3 ) -
                     & left(  0,  0,  0,  0 );
      if ( j <= offset || left_alloc <= j ) { update |= 1 ; }
      offset = j ;
    }

    offset = -1 ;
    for ( unsigned i0 = 0 ; i0 < unsigned(right.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(right.dimension_1()) ; ++i1 )
    for ( unsigned i2 = 0 ; i2 < unsigned(right.dimension_2()) ; ++i2 )
    for ( unsigned i3 = 0 ; i3 < unsigned(right.dimension_3()) ; ++i3 )
    {
      const long j = & right( i0, i1, i2, i3 ) -
                     & right(  0,  0,  0,  0 );
      if ( j <= offset || right_alloc <= j ) { update |= 2 ; }
      offset = j ;
    }
  }
};

template< class DataType , class DeviceType >
struct TestViewOperator_LeftAndRight< DataType , DeviceType , 3 >
{
  typedef DeviceType                          execution_space ;
  typedef typename execution_space::memory_space  memory_space ;
  typedef typename execution_space::size_type     size_type ;

  typedef int value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    const volatile value_type & input )
    { update |= input ; }

  KOKKOS_INLINE_FUNCTION
  static void init( value_type & update )
    { update = 0 ; }


  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutLeft, execution_space > left_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutRight, execution_space > right_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutStride, execution_space > stride_view ;

  left_view    left ;
  right_view   right ;
  stride_view  left_stride ;
  stride_view  right_stride ;
  long         left_alloc ;
  long         right_alloc ;

  TestViewOperator_LeftAndRight(unsigned N0, unsigned N1, unsigned N2)
    : left(  std::string("left") , N0, N1, N2 )
    , right( std::string("right") , N0, N1, N2 )
    , left_stride( left )
    , right_stride( right )
    , left_alloc( allocation_count( left ) )
    , right_alloc( allocation_count( right ) )
    {}

  static void testit(unsigned N0, unsigned N1, unsigned N2)
  {
    TestViewOperator_LeftAndRight driver (N0, N1, N2);

    int error_flag = 0 ;

    Kokkos::parallel_reduce( 1 , driver , error_flag );

    ASSERT_EQ( error_flag , 0 );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type , value_type & update ) const
  {
    long offset ;

    offset = -1 ;
    for ( unsigned i2 = 0 ; i2 < unsigned(left.dimension_2()) ; ++i2 )
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    {
      const long j = & left( i0, i1, i2 ) -
                     & left(  0,  0,  0 );
      if ( j <= offset || left_alloc <= j ) { update |= 1 ; }
      offset = j ;

      if ( & left(i0,i1,i2) != & left_stride(i0,i1,i2) ) { update |= 4 ; }
    }

    offset = -1 ;
    for ( unsigned i0 = 0 ; i0 < unsigned(right.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(right.dimension_1()) ; ++i1 )
    for ( unsigned i2 = 0 ; i2 < unsigned(right.dimension_2()) ; ++i2 )
    {
      const long j = & right( i0, i1, i2 ) -
                     & right(  0,  0,  0 );
      if ( j <= offset || right_alloc <= j ) { update |= 2 ; }
      offset = j ;

      if ( & right(i0,i1,i2) != & right_stride(i0,i1,i2) ) { update |= 8 ; }
    }

    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    for ( unsigned i2 = 0 ; i2 < unsigned(left.dimension_2()) ; ++i2 )
    {
      if ( & left(i0,i1,i2)  != & left(i0,i1,i2,0,0,0,0,0) )  { update |= 3 ; }
      if ( & right(i0,i1,i2) != & right(i0,i1,i2,0,0,0,0,0) ) { update |= 3 ; }
    }
  }
};

template< class DataType , class DeviceType >
struct TestViewOperator_LeftAndRight< DataType , DeviceType , 2 >
{
  typedef DeviceType                          execution_space ;
  typedef typename execution_space::memory_space  memory_space ;
  typedef typename execution_space::size_type     size_type ;

  typedef int value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    const volatile value_type & input )
    { update |= input ; }

  KOKKOS_INLINE_FUNCTION
  static void init( value_type & update )
    { update = 0 ; }


  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutLeft, execution_space > left_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutRight, execution_space > right_view ;

  left_view    left ;
  right_view   right ;
  long         left_alloc ;
  long         right_alloc ;

  TestViewOperator_LeftAndRight(unsigned N0, unsigned N1)
    : left(  "left" , N0, N1 )
    , right( "right" , N0, N1 )
    , left_alloc( allocation_count( left ) )
    , right_alloc( allocation_count( right ) )
    {}

  static void testit(unsigned N0, unsigned N1)
  {
    TestViewOperator_LeftAndRight driver(N0, N1);

    int error_flag = 0 ;

    Kokkos::parallel_reduce( 1 , driver , error_flag );

    ASSERT_EQ( error_flag , 0 );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type , value_type & update ) const
  {
    long offset ;

    offset = -1 ;
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    {
      const long j = & left( i0, i1 ) -
                     & left(  0,  0 );
      if ( j <= offset || left_alloc <= j ) { update |= 1 ; }
      offset = j ;
    }

    offset = -1 ;
    for ( unsigned i0 = 0 ; i0 < unsigned(right.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(right.dimension_1()) ; ++i1 )
    {
      const long j = & right( i0, i1 ) -
                     & right(  0,  0 );
      if ( j <= offset || right_alloc <= j ) { update |= 2 ; }
      offset = j ;
    }

    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    for ( unsigned i1 = 0 ; i1 < unsigned(left.dimension_1()) ; ++i1 )
    {
      if ( & left(i0,i1)  != & left(i0,i1,0,0,0,0,0,0) )  { update |= 3 ; }
      if ( & right(i0,i1) != & right(i0,i1,0,0,0,0,0,0) ) { update |= 3 ; }
    }
  }
};

template< class DataType , class DeviceType >
struct TestViewOperator_LeftAndRight< DataType , DeviceType , 1 >
{
  typedef DeviceType                          execution_space ;
  typedef typename execution_space::memory_space  memory_space ;
  typedef typename execution_space::size_type     size_type ;

  typedef int value_type ;

  KOKKOS_INLINE_FUNCTION
  static void join( volatile value_type & update ,
                    const volatile value_type & input )
    { update |= input ; }

  KOKKOS_INLINE_FUNCTION
  static void init( value_type & update )
    { update = 0 ; }


  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutLeft, execution_space > left_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutRight, execution_space > right_view ;

  typedef Kokkos::
    Experimental::DynRankView< DataType, Kokkos::LayoutStride, execution_space > stride_view ;

  left_view    left ;
  right_view   right ;
  stride_view  left_stride ;
  stride_view  right_stride ;
  long         left_alloc ;
  long         right_alloc ;

  TestViewOperator_LeftAndRight(unsigned N0)
    : left(  "left" , N0 )
    , right( "right" , N0 )
    , left_stride( left )
    , right_stride( right )
    , left_alloc( allocation_count( left ) )
    , right_alloc( allocation_count( right ) )
    {}

  static void testit(unsigned N0)
  {
    TestViewOperator_LeftAndRight driver (N0) ;

    int error_flag = 0 ;

    Kokkos::parallel_reduce( 1 , driver , error_flag );

    ASSERT_EQ( error_flag , 0 );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type , value_type & update ) const
  {
    for ( unsigned i0 = 0 ; i0 < unsigned(left.dimension_0()) ; ++i0 )
    {
      if ( & left(i0)  != & left(i0,0,0,0,0,0,0,0) )  { update |= 3 ; }
      if ( & right(i0) != & right(i0,0,0,0,0,0,0,0) ) { update |= 3 ; }
      if ( & left(i0)  != & left_stride(i0) ) { update |= 4 ; }
      if ( & right(i0) != & right_stride(i0) ) { update |= 8 ; }
    }
  }
};

/*--------------------------------------------------------------------------*/

template< typename T, class DeviceType >
class TestDynViewAPI
{
public:
  typedef DeviceType        device ;

  enum { N0 = 1000 ,
         N1 = 3 ,
         N2 = 5 ,
         N3 = 7 };

  typedef Kokkos::Experimental::DynRankView< T , device > dView0 ;
  typedef Kokkos::Experimental::DynRankView< const T , device > const_dView0 ;

  typedef Kokkos::Experimental::DynRankView< T, device, Kokkos::MemoryUnmanaged > dView0_unmanaged ;
  typedef typename dView0::host_mirror_space host ;

  TestDynViewAPI()
  {
    run_test_mirror();
    run_test();
    run_test_scalar();
    run_test_const();
    run_test_subview();
    run_test_subview_strided();
    run_test_vector();

    TestViewOperator< T , device >::testit();
    TestViewOperator_LeftAndRight< int , device , 8 >::testit(2,3,4,2,3,4,2,3); 
    TestViewOperator_LeftAndRight< int , device , 7 >::testit(2,3,4,2,3,4,2); 
    TestViewOperator_LeftAndRight< int , device , 6 >::testit(2,3,4,2,3,4); 
    TestViewOperator_LeftAndRight< int , device , 5 >::testit(2,3,4,2,3);
    TestViewOperator_LeftAndRight< int , device , 4 >::testit(2,3,4,2);
    TestViewOperator_LeftAndRight< int , device , 3 >::testit(2,3,4);
    TestViewOperator_LeftAndRight< int , device , 2 >::testit(2,3);
    TestViewOperator_LeftAndRight< int , device , 1 >::testit(2);
  }

  static void run_test_mirror()
  {
    typedef Kokkos::Experimental::DynRankView< int , host > view_type ;
    typedef typename view_type::HostMirror mirror_type ;
    view_type a("a");
    mirror_type am = Kokkos::Experimental::create_mirror_view(a);
    mirror_type ax = Kokkos::Experimental::create_mirror(a);
    ASSERT_EQ( & a() , & am() );
  }

  static void run_test_scalar()
  {
    typedef typename dView0::HostMirror  hView0 ;

    dView0 dx , dy ;
    hView0 hx , hy ;

    dx = dView0( "dx" );
    dy = dView0( "dy" );

    hx = Kokkos::Experimental::create_mirror( dx );
    hy = Kokkos::Experimental::create_mirror( dy );

    hx() = 1 ;

    Kokkos::Experimental::deep_copy( dx , hx );
    Kokkos::Experimental::deep_copy( dy , dx );
    Kokkos::Experimental::deep_copy( hy , dy );

    ASSERT_EQ( hx(), hy() );
  }

  static void run_test()
  {
    // mfh 14 Feb 2014: This test doesn't actually create instances of
    // these types.  In order to avoid "declared but unused typedef"
    // warnings, we declare empty instances of these types, with the
    // usual "(void)" marker to avoid compiler warnings for unused
    // variables.

    typedef typename dView0::HostMirror  hView0 ;

    {
      hView0 thing;
      (void) thing;
    }

    dView0 dx , dy , dz ;
    hView0 hx , hy , hz ;

    ASSERT_TRUE( dx.ptr_on_device() == 0 );
    ASSERT_TRUE( dy.ptr_on_device() == 0 );
    ASSERT_TRUE( dz.ptr_on_device() == 0 );
    ASSERT_TRUE( hx.ptr_on_device() == 0 );
    ASSERT_TRUE( hy.ptr_on_device() == 0 );
    ASSERT_TRUE( hz.ptr_on_device() == 0 );
    ASSERT_EQ( dx.dimension_0() , 0u );
    ASSERT_EQ( dy.dimension_0() , 0u );
    ASSERT_EQ( dz.dimension_0() , 0u );
    ASSERT_EQ( hx.dimension_0() , 0u );
    ASSERT_EQ( hy.dimension_0() , 0u );
    ASSERT_EQ( hz.dimension_0() , 0u );
    ASSERT_EQ( dx.rank() , 0u );
    ASSERT_EQ( hx.rank() , 0u );

    dx = dView0( "dx" , N1 , N2 , N3 );
    dy = dView0( "dy" , N1 , N2 , N3 );

    hx = hView0( "hx" , N1 , N2 , N3 );
    hy = hView0( "hy" , N1 , N2 , N3 );

    ASSERT_EQ( dx.dimension_0() , unsigned(N1) );
    ASSERT_EQ( dy.dimension_0() , unsigned(N1) );
    ASSERT_EQ( hx.dimension_0() , unsigned(N1) );
    ASSERT_EQ( hy.dimension_0() , unsigned(N1) );
    ASSERT_EQ( dx.rank() , 3 );
    ASSERT_EQ( hx.rank() , 3 );

    dx = dView0( "dx" , N0 , N1 , N2 , N3 );
    dy = dView0( "dy" , N0 , N1 , N2 , N3 );
    hx = hView0( "hx" , N0 , N1 , N2 , N3 );
    hy = hView0( "hy" , N0 , N1 , N2 , N3 );

    ASSERT_EQ( dx.dimension_0() , unsigned(N0) );
    ASSERT_EQ( dy.dimension_0() , unsigned(N0) );
    ASSERT_EQ( hx.dimension_0() , unsigned(N0) );
    ASSERT_EQ( hy.dimension_0() , unsigned(N0) );
    ASSERT_EQ( dx.rank() , 4 );
    ASSERT_EQ( hx.rank() , 4 );

    ASSERT_EQ( dx.use_count() , size_t(1) );

    dView0_unmanaged unmanaged_dx = dx;
    ASSERT_EQ( dx.use_count() , size_t(1) );

    dView0_unmanaged unmanaged_from_ptr_dx = dView0_unmanaged(dx.ptr_on_device(),
                                                              dx.dimension_0(),
                                                              dx.dimension_1(),
                                                              dx.dimension_2(),
                                                              dx.dimension_3());

    {
      // Destruction of this view should be harmless
      const_dView0 unmanaged_from_ptr_const_dx( dx.ptr_on_device() ,
                                                dx.dimension_0() ,
                                                dx.dimension_1() ,
                                                dx.dimension_2() ,
                                                dx.dimension_3() );
    }

    const_dView0 const_dx = dx ;
    ASSERT_EQ( dx.use_count() , size_t(2) );

    {
      const_dView0 const_dx2;
      const_dx2 = const_dx;
      ASSERT_EQ( dx.use_count() , size_t(3) );

      const_dx2 = dy;
      ASSERT_EQ( dx.use_count() , size_t(2) );

      const_dView0 const_dx3(dx);
      ASSERT_EQ( dx.use_count() , size_t(3) );
      
      dView0_unmanaged dx4_unmanaged(dx);
      ASSERT_EQ( dx.use_count() , size_t(3) );
    }

    ASSERT_EQ( dx.use_count() , size_t(2) );


    ASSERT_FALSE( dx.ptr_on_device() == 0 );
    ASSERT_FALSE( const_dx.ptr_on_device() == 0 );
    ASSERT_FALSE( unmanaged_dx.ptr_on_device() == 0 );
    ASSERT_FALSE( unmanaged_from_ptr_dx.ptr_on_device() == 0 );
    ASSERT_FALSE( dy.ptr_on_device() == 0 );
    ASSERT_NE( dx , dy );

    ASSERT_EQ( dx.dimension_0() , unsigned(N0) );
    ASSERT_EQ( dx.dimension_1() , unsigned(N1) );
    ASSERT_EQ( dx.dimension_2() , unsigned(N2) );
    ASSERT_EQ( dx.dimension_3() , unsigned(N3) );

    ASSERT_EQ( dy.dimension_0() , unsigned(N0) );
    ASSERT_EQ( dy.dimension_1() , unsigned(N1) );
    ASSERT_EQ( dy.dimension_2() , unsigned(N2) );
    ASSERT_EQ( dy.dimension_3() , unsigned(N3) );

    ASSERT_EQ( unmanaged_from_ptr_dx.capacity(),unsigned(N0)*unsigned(N1)*unsigned(N2)*unsigned(N3) );

    hx = Kokkos::Experimental::create_mirror( dx );
    hy = Kokkos::Experimental::create_mirror( dy );

    // T v1 = hx() ;    // Generates compile error as intended
    // T v2 = hx(0,0) ; // Generates compile error as intended
    // hx(0,0) = v2 ;   // Generates compile error as intended
/*
#if ! KOKKOS_USING_EXP_VIEW
    // Testing with asynchronous deep copy with respect to device
    {
      size_t count = 0 ;
      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < hx.dimension_1() ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < hx.dimension_2() ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < hx.dimension_3() ; ++i3 ) {
        hx(ip,i1,i2,i3) = ++count ;
      }}}}


      Kokkos::deep_copy(typename hView0::execution_space(), dx , hx );
      Kokkos::deep_copy(typename hView0::execution_space(), dy , dx );
      Kokkos::deep_copy(typename hView0::execution_space(), hy , dy );

      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
        { ASSERT_EQ( hx(ip,i1,i2,i3) , hy(ip,i1,i2,i3) ); }
      }}}}

      Kokkos::deep_copy(typename hView0::execution_space(), dx , T(0) );
      Kokkos::deep_copy(typename hView0::execution_space(), hx , dx );

      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
        { ASSERT_EQ( hx(ip,i1,i2,i3) , T(0) ); }
      }}}}
    }

    // Testing with asynchronous deep copy with respect to host
    {
      size_t count = 0 ;
      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < hx.dimension_1() ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < hx.dimension_2() ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < hx.dimension_3() ; ++i3 ) {
        hx(ip,i1,i2,i3) = ++count ;
      }}}}

      Kokkos::deep_copy(typename dView0::execution_space(), dx , hx );
      Kokkos::deep_copy(typename dView0::execution_space(), dy , dx );
      Kokkos::deep_copy(typename dView0::execution_space(), hy , dy );

      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
        { ASSERT_EQ( hx(ip,i1,i2,i3) , hy(ip,i1,i2,i3) ); }
      }}}}

      Kokkos::deep_copy(typename dView0::execution_space(), dx , T(0) );
      Kokkos::deep_copy(typename dView0::execution_space(), hx , dx );

      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
        { ASSERT_EQ( hx(ip,i1,i2,i3) , T(0) ); }
      }}}}
    }
#endif */ // #if ! KOKKOS_USING_EXP_VIEW

    // Testing with synchronous deep copy
    {
      size_t count = 0 ;
      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < hx.dimension_1() ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < hx.dimension_2() ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < hx.dimension_3() ; ++i3 ) {
        hx(ip,i1,i2,i3) = ++count ;
      }}}}

      Kokkos::Experimental::deep_copy( dx , hx );
      Kokkos::Experimental::deep_copy( dy , dx );
      Kokkos::Experimental::deep_copy( hy , dy );

      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
        { ASSERT_EQ( hx(ip,i1,i2,i3) , hy(ip,i1,i2,i3) ); }
      }}}}

      Kokkos::Experimental::deep_copy( dx , T(0) );
      Kokkos::Experimental::deep_copy( hx , dx );

      for ( size_t ip = 0 ; ip < N0 ; ++ip ) {
      for ( size_t i1 = 0 ; i1 < N1 ; ++i1 ) {
      for ( size_t i2 = 0 ; i2 < N2 ; ++i2 ) {
      for ( size_t i3 = 0 ; i3 < N3 ; ++i3 ) {
        { ASSERT_EQ( hx(ip,i1,i2,i3) , T(0) ); }
      }}}}
    }
    dz = dx ; ASSERT_EQ( dx, dz); ASSERT_NE( dy, dz);
    dz = dy ; ASSERT_EQ( dy, dz); ASSERT_NE( dx, dz);

    dx = dView0();
    ASSERT_TRUE( dx.ptr_on_device() == 0 );
    ASSERT_FALSE( dy.ptr_on_device() == 0 );
    ASSERT_FALSE( dz.ptr_on_device() == 0 );
    dy = dView0();
    ASSERT_TRUE( dx.ptr_on_device() == 0 );
    ASSERT_TRUE( dy.ptr_on_device() == 0 );
    ASSERT_FALSE( dz.ptr_on_device() == 0 );
    dz = dView0();
    ASSERT_TRUE( dx.ptr_on_device() == 0 );
    ASSERT_TRUE( dy.ptr_on_device() == 0 );
    ASSERT_TRUE( dz.ptr_on_device() == 0 );
  }

  typedef T DataType ;

  static void
  check_auto_conversion_to_const(
     const Kokkos::Experimental::DynRankView< const DataType , device > & arg_const ,
     const Kokkos::Experimental::DynRankView< DataType , device > & arg )
  {
    ASSERT_TRUE( arg_const == arg );
  }

  static void run_test_const()
  {
    typedef Kokkos::Experimental::DynRankView< DataType , device > typeX ;
    typedef Kokkos::Experimental::DynRankView< const DataType , device > const_typeX ;
    typedef Kokkos::Experimental::DynRankView< const DataType , device , Kokkos::MemoryRandomAccess > const_typeR ;
    typeX x( "X", 2 );
    const_typeX xc = x ;
    const_typeR xr = x ;

    ASSERT_TRUE( xc == x );
    ASSERT_TRUE( x == xc );

    // For CUDA the constant random access View does not return
    // an lvalue reference due to retrieving through texture cache
    // therefore not allowed to query the underlying pointer.
#if defined(KOKKOS_HAVE_CUDA)
    if ( ! std::is_same< typename device::execution_space , Kokkos::Cuda >::value )
#endif
    {
      ASSERT_TRUE( x.ptr_on_device() == xr.ptr_on_device() );
    }

    // typeX xf = xc ; // setting non-const from const must not compile

    check_auto_conversion_to_const( x , x );
  }


  static void run_test_subview()
  {
// LayoutStride may be causing issues with how it is declared...

    typedef Kokkos::Experimental::DynRankView< const T , device > sView ;
    typedef Kokkos::Experimental::DynRankView< T , Kokkos::LayoutStride , device > sdView ; //this works

    dView0 d0( "d0" );
/*
    std::cout << " d0 rank " << d0.rank() <<std::endl;
    std::cout << " d0 dim0 "<< d0.dimension_0() <<std::endl;
    std::cout << " d0 dim1 "<< d0.dimension_1() <<std::endl;
    std::cout << " d0 dim2 "<< d0.dimension_2() <<std::endl;
    std::cout << " d0 dim3 "<< d0.dimension_3() <<std::endl;
    std::cout << " d0 dim4 "<< d0.dimension_4() <<std::endl;
    std::cout << " d0 dim5 "<< d0.dimension_5() <<std::endl;
    std::cout << " d0 dim6 "<< d0.dimension_6() <<std::endl;
    std::cout << " d0 dim7 "<< d0.dimension_7() <<std::endl;
*/

    unsigned order[] = { 6,5,4,3,2,1,0 }, dimen[] = { N0, N1, N2, 2, 2, 2, 2 };
    sdView d7( "d7", Kokkos::LayoutStride::order_dimensions(7, order, dimen) );
/*
    std::cout << " d7 rank " << d7.rank() <<std::endl;
    std::cout << " d7 dim0 "<< d7.dimension_0() <<std::endl;
    std::cout << " d7 dim1 "<< d7.dimension_1() <<std::endl;
    std::cout << " d7 dim2 "<< d7.dimension_2() <<std::endl;
    std::cout << " d7 dim3 "<< d7.dimension_3() <<std::endl;
    std::cout << " d7 dim4 "<< d7.dimension_4() <<std::endl;
    std::cout << " d7 dim5 "<< d7.dimension_5() <<std::endl;
    std::cout << " d7 dim6 "<< d7.dimension_6() <<std::endl;
    std::cout << " d7 dim7 "<< d7.dimension_7() <<std::endl;
*/

    unsigned order2[] = { 7,6,5,4,3,2,1,0 }, dimen2[] = { N0, N1, N2, 2, 2, 2, 2, 2 };
    sdView d8( "d8", Kokkos::LayoutStride::order_dimensions(8, order2, dimen2) );
/*
    std::cout << " d8 rank " << d8.rank() <<std::endl;
    std::cout << " d8 dim0 "<< d8.dimension_0() <<std::endl;
    std::cout << " d8 dim1 "<< d8.dimension_1() <<std::endl;
    std::cout << " d8 dim2 "<< d8.dimension_2() <<std::endl;
    std::cout << " d8 dim3 "<< d8.dimension_3() <<std::endl;
    std::cout << " d8 dim4 "<< d8.dimension_4() <<std::endl;
    std::cout << " d8 dim5 "<< d8.dimension_5() <<std::endl;
    std::cout << " d8 dim6 "<< d8.dimension_6() <<std::endl;
    std::cout << " d8 dim7 "<< d8.dimension_7() <<std::endl;
*/

    unsigned order3[] = { 4,3,2,1,0 }, dimen3[] = { N0, N1, N2, 2, 2 };
    sdView d5( "d5", Kokkos::LayoutStride::order_dimensions(5, order3, dimen3) );
/*
    std::cout << " d5 rank " << d5.rank() <<std::endl;
    std::cout << " d5 dim0 "<< d5.dimension_0() <<std::endl;
    std::cout << " d5 dim1 "<< d5.dimension_1() <<std::endl;
    std::cout << " d5 dim2 "<< d5.dimension_2() <<std::endl;
    std::cout << " d5 dim3 "<< d5.dimension_3() <<std::endl;
    std::cout << " d5 dim4 "<< d5.dimension_4() <<std::endl;
    std::cout << " d5 dim5 "<< d5.dimension_5() <<std::endl;
    std::cout << " d5 dim6 "<< d5.dimension_6() <<std::endl;
    std::cout << " d5 dim7 "<< d5.dimension_7() <<std::endl;
*/

    dView0 dtest("dtest" , N0 , N1 , N2 , 2 , 2 , 2 , 2 , 2);

    sView s0 = d0 ;
    sdView ds0 = Kokkos::Experimental::subdynrankview( d8 , 1,1,1,1,1,1,1,1); //Should be rank0 subview

//Basic test - ALL
    sdView ds1 = Kokkos::Experimental::subdynrankview( d8 , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() ); //compiles and runs
//  Send a single value for one rank
    sdView ds2 = Kokkos::Experimental::subdynrankview( d8 , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , 1 );
//  Send a std::pair as a rank
    sdView ds3 = Kokkos::Experimental::subdynrankview( d8 , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , std::pair<unsigned,unsigned>(1,2) );
//  Send a kokkos::pair as a rank
    std::cout<<" dt8 rank 8 subview... "<<std::endl; 
    sdView dt8 = Kokkos::Experimental::subdynrankview( dtest , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::pair<unsigned,unsigned>(0,1) );

    std::cout<<" ds8 rank 8 subview... "<<std::endl;
    sdView ds8 = Kokkos::Experimental::subdynrankview( d8 , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::pair<unsigned,unsigned>(0,1) );

// Breaks...
    std::cout<<" ds7 rank 7 subview... "<<std::endl;
    sdView ds7 = Kokkos::Experimental::subdynrankview( d7 , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::pair<unsigned,unsigned>(0,1) );

    std::cout<<" ds5 rank 5 subview... "<<std::endl;
    sdView ds5 = Kokkos::Experimental::subdynrankview( d5 , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::ALL() , Kokkos::pair<unsigned,unsigned>(0,1) );
  }

  static void run_test_subview_strided()
  {
    typedef Kokkos::Experimental::DynRankView < int , Kokkos::LayoutLeft , host > drview_left ;
    typedef Kokkos::Experimental::DynRankView < int , Kokkos::LayoutRight , host > drview_right ;
    typedef Kokkos::Experimental::DynRankView < int , Kokkos::LayoutStride , host > drview_stride ;

    drview_left  xl2( "xl2", 100 , 200 );
    drview_right xr2( "xr2", 100 , 200 );
    drview_stride yl1 = Kokkos::Experimental::subdynrankview( xl2 , 0 , Kokkos::ALL() );
    drview_stride yl2 = Kokkos::Experimental::subdynrankview( xl2 , 1 , Kokkos::ALL() );
    drview_stride ys1 = Kokkos::Experimental::subdynrankview( xr2 , 0 , Kokkos::ALL() );
    drview_stride ys2 = Kokkos::Experimental::subdynrankview( xr2 , 1 , Kokkos::ALL() );
    drview_stride yr1 = Kokkos::Experimental::subdynrankview( xr2 , 0 , Kokkos::ALL() ); //errors out
    drview_stride yr2 = Kokkos::Experimental::subdynrankview( xr2 , 1 , Kokkos::ALL() );

// Original subview testing
/*
//    typedef Kokkos::Experimental::View < int[100][200] , Kokkos::LayoutRight , host > view_right_2 ;
    typedef Kokkos::Experimental::View < int[100][200] , Kokkos::LayoutLeft , host > view_left_2 ;

//    typedef Kokkos::Experimental::View < int* , host > sview_1 ;
    typedef Kokkos::Experimental::View < int* , Kokkos::LayoutStride , host > strview_1 ;

    view_left_2 vl2("vl2");
//    sview_1 svl1 = Kokkos::Experimental::subview( vl2 , 0 , Kokkos::ALL() ); //breaks
    strview_1 stvl1 = Kokkos::Experimental::subview( vl2 , 0 , Kokkos::ALL() );

    std::cout << "  vl2.dim0 " << vl2.dimension_0() <<std::endl;
    std::cout << "  vl2.dim1 " << vl2.dimension_1() <<std::endl;
    std::cout << "  stvl1.dim0 " << stvl1.dimension_0() <<std::endl;//should be 200
    std::cout << "  stvl1.dim1 " << stvl1.dimension_1() <<std::endl;//should be 1

    typedef Kokkos::Experimental::View < int[20][5][3] , Kokkos::LayoutRight , host > view_right_3 ;
    typedef Kokkos::Experimental::View < int** , host > view_right_2 ;

    view_right_3 vr3("vr3");
    for ( int i = 0 ; i < 20 ; ++i ) {
    for ( int j = 0 ; j < 5 ; ++j ) {
    for ( int k = 0 ; k < 3 ; ++k ) {
      vr3(i,j,k) = k + j*3 + i *15;
    }}}

    view_right_2 sv2 = Kokkos::Experimental::subview(vr3 , Kokkos::ALL() , 1 , Kokkos::ALL() );

    std::cout << "  vr3.dim0 " << vr3.dimension_0() <<std::endl;
    std::cout << "  vr3.dim1 " << vr3.dimension_1() <<std::endl;
    std::cout << "  vr3.dim2 " << vr3.dimension_2() <<std::endl;
    std::cout << "  sv2.dim0 " << sv2.dimension_0() <<std::endl;
    std::cout << "  sv2.dim1 " << sv2.dimension_1() <<std::endl;
    std::cout << "  sv2.dim2 " << sv2.dimension_2() <<std::endl;
    std::cout << "\n";
*/

// end original subview testing

    std::cout << "  xl2.dim0 " << xl2.dimension_0() <<std::endl;
    std::cout << "  xl2.dim1 " << xl2.dimension_1() <<std::endl;
    std::cout << "  yl1.dim0 " << yl1.dimension_0() <<std::endl; //should be 200
    std::cout << "  yl1.dim1 " << yl1.dimension_1() <<std::endl; //should be 1
    std::cout << "  xr2.dim0 " << xr2.dimension_0() <<std::endl;
    std::cout << "  xr2.dim1 " << xr2.dimension_1() <<std::endl;
//    std::cout << "  yr1.dim0 " << yr1.dimension_0() <<std::endl;
//    std::cout << "  yr1.dim1 " << yr1.dimension_1() <<std::endl;
//    std::cout << "  yr2.dim0 " << yr2.dimension_0() <<std::endl;
//    std::cout << "  yr2.dim1 " << yr2.dimension_1() <<std::endl;
    ASSERT_EQ( yl1.dimension_0() , xl2.dimension_1() );
    ASSERT_EQ( yl2.dimension_0() , xl2.dimension_1() );

    ASSERT_EQ( yr1.dimension_0() , xr2.dimension_1() );
    ASSERT_EQ( yr2.dimension_0() , xr2.dimension_1() );

    ASSERT_EQ( & yl1(0) - & xl2(0,0) , 0 );
    ASSERT_EQ( & yl2(0) - & xl2(1,0) , 0 );
    ASSERT_EQ( & yr1(0) - & xr2(0,0) , 0 );
    ASSERT_EQ( & yr2(0) - & xr2(1,0) , 0 );


    drview_left  xl4( "xl4", 10 , 20 , 30 , 40 );
    drview_right xr4( "xr4", 10 , 20 , 30 , 40 );

    drview_stride yl4 = Kokkos::Experimental::subdynrankview( xl4 , 1 , Kokkos::ALL() , 2 , Kokkos::ALL() );
    drview_stride yr4 = Kokkos::Experimental::subdynrankview( xr4 , 1 , Kokkos::ALL() , 2 , Kokkos::ALL() );

    ASSERT_EQ( yl4.dimension_0() , xl4.dimension_1() );
    ASSERT_EQ( yl4.dimension_1() , xl4.dimension_3() );
    ASSERT_EQ( yr4.dimension_0() , xr4.dimension_1() );
    ASSERT_EQ( yr4.dimension_1() , xr4.dimension_3() );

    ASSERT_EQ( & yl4(4,4) - & xl4(1,4,2,4) , 0 );
    ASSERT_EQ( & yr4(4,4) - & xr4(1,4,2,4) , 0 );

  }

  static void run_test_vector()
  {
    static const unsigned Length = 1000 , Count = 8 ;

    typedef typename Kokkos::Experimental::DynRankView< T , Kokkos::LayoutLeft , host > vector_type ; //rank 1
    typedef typename Kokkos::Experimental::DynRankView< T , Kokkos::LayoutLeft , host > multivector_type ; //rank 2

    typedef typename Kokkos::Experimental::DynRankView< T , Kokkos::LayoutRight , host > vector_right_type ; //rank 1
    typedef typename Kokkos::Experimental::DynRankView< T , Kokkos::LayoutRight , host > multivector_right_type ; //rank 2

    typedef typename Kokkos::Experimental::DynRankView< const T , Kokkos::LayoutRight , host > const_vector_right_type ; //rank 1
    typedef typename Kokkos::Experimental::DynRankView< const T , Kokkos::LayoutLeft , host > const_vector_type ; //rank 1
    typedef typename Kokkos::Experimental::DynRankView< const T , Kokkos::LayoutLeft , host > const_multivector_type ; //rank 2

    multivector_type mv = multivector_type( "mv" , Length , Count );
    multivector_right_type mv_right = multivector_right_type( "mv" , Length , Count );

    typedef typename Kokkos::Experimental::DynRankView< T , Kokkos::LayoutStride , host > svector_type ;
    typedef typename Kokkos::Experimental::DynRankView< T , Kokkos::LayoutStride , host > smultivector_type ;
    typedef typename Kokkos::Experimental::DynRankView< const T , Kokkos::LayoutStride , host > const_svector_right_type ;
    typedef typename Kokkos::Experimental::DynRankView< const T , Kokkos::LayoutStride , host > const_svector_type ;

    svector_type v1 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 0 );
    svector_type v2 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 1 );
    svector_type v3 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 2 );

    svector_type rv1 = Kokkos::Experimental::subdynrankview( mv_right , 0 , Kokkos::ALL() );
    svector_type rv2 = Kokkos::Experimental::subdynrankview( mv_right , 1 , Kokkos::ALL() );
    svector_type rv3 = Kokkos::Experimental::subdynrankview( mv_right , 2 , Kokkos::ALL() );

    smultivector_type mv1 = Kokkos::Experimental::subdynrankview( mv , std::make_pair( 1 , 998 ) ,
                                                 std::make_pair( 2 , 5 ) );

    smultivector_type mvr1 =
      Kokkos::Experimental::subdynrankview( mv_right ,
                       std::make_pair( 1 , 998 ) ,
                       std::make_pair( 2 , 5 ) );

    const_svector_type cv1 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL(), 0 );
    const_svector_type cv2 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL(), 1 );
    const_svector_type cv3 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL(), 2 );

    svector_type vr1 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 0 );
    svector_type vr2 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 1 );
    svector_type vr3 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 2 );

    const_svector_right_type cvr1 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 0 );
    const_svector_right_type cvr2 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 1 );
    const_svector_right_type cvr3 = Kokkos::Experimental::subdynrankview( mv , Kokkos::ALL() , 2 );


    ASSERT_TRUE( & v1[0] == & v1(0) );
    ASSERT_TRUE( & v1[0] == & mv(0,0) );
    ASSERT_TRUE( & v2[0] == & mv(0,1) );
    ASSERT_TRUE( & v3[0] == & mv(0,2) );

    ASSERT_TRUE( & cv1[0] == & mv(0,0) );
    ASSERT_TRUE( & cv2[0] == & mv(0,1) );
    ASSERT_TRUE( & cv3[0] == & mv(0,2) );

    ASSERT_TRUE( & vr1[0] == & mv(0,0) );
    ASSERT_TRUE( & vr2[0] == & mv(0,1) );
    ASSERT_TRUE( & vr3[0] == & mv(0,2) );

    ASSERT_TRUE( & cvr1[0] == & mv(0,0) );
    ASSERT_TRUE( & cvr2[0] == & mv(0,1) );
    ASSERT_TRUE( & cvr3[0] == & mv(0,2) );


    ASSERT_TRUE( & mv1(0,0) == & mv( 1 , 2 ) );
    ASSERT_TRUE( & mv1(1,1) == & mv( 2 , 3 ) );
    ASSERT_TRUE( & mv1(3,2) == & mv( 4 , 4 ) );
    ASSERT_TRUE( & mvr1(0,0) == & mv_right( 1 , 2 ) );
    ASSERT_TRUE( & mvr1(1,1) == & mv_right( 2 , 3 ) );
    ASSERT_TRUE( & mvr1(3,2) == & mv_right( 4 , 4 ) );
/*

    typedef Kokkos::View< T* ,  Kokkos::LayoutLeft , host > vector_type ;
    typedef Kokkos::View< T** , Kokkos::LayoutLeft , host > multivector_type ;

    typedef Kokkos::View< T* ,  Kokkos::LayoutRight , host > vector_right_type ;
    typedef Kokkos::View< T** , Kokkos::LayoutRight , host > multivector_right_type ;

    typedef Kokkos::View< const T* , Kokkos::LayoutRight, host > const_vector_right_type ;
    typedef Kokkos::View< const T* , Kokkos::LayoutLeft , host > const_vector_type ;
    typedef Kokkos::View< const T** , Kokkos::LayoutLeft , host > const_multivector_type ;

    multivector_type mv = multivector_type( "mv" , Length , Count );
    multivector_right_type mv_right = multivector_right_type( "mv" , Length , Count );

    vector_type v1 = Kokkos::subview( mv , Kokkos::ALL() , 0 );
    vector_type v2 = Kokkos::subview( mv , Kokkos::ALL() , 1 );
    vector_type v3 = Kokkos::subview( mv , Kokkos::ALL() , 2 );

    vector_type rv1 = Kokkos::subview( mv_right , 0 , Kokkos::ALL() );
    vector_type rv2 = Kokkos::subview( mv_right , 1 , Kokkos::ALL() );
    vector_type rv3 = Kokkos::subview( mv_right , 2 , Kokkos::ALL() );

    multivector_type mv1 = Kokkos::subview( mv , std::make_pair( 1 , 998 ) ,
                                                 std::make_pair( 2 , 5 ) );

    multivector_right_type mvr1 =
      Kokkos::subview( mv_right ,
                       std::make_pair( 1 , 998 ) ,
                       std::make_pair( 2 , 5 ) );

    const_vector_type cv1 = Kokkos::subview( mv , Kokkos::ALL(), 0 );
    const_vector_type cv2 = Kokkos::subview( mv , Kokkos::ALL(), 1 );
    const_vector_type cv3 = Kokkos::subview( mv , Kokkos::ALL(), 2 );

    vector_right_type vr1 = Kokkos::subview( mv , Kokkos::ALL() , 0 );
    vector_right_type vr2 = Kokkos::subview( mv , Kokkos::ALL() , 1 );
    vector_right_type vr3 = Kokkos::subview( mv , Kokkos::ALL() , 2 );

    const_vector_right_type cvr1 = Kokkos::subview( mv , Kokkos::ALL() , 0 );
    const_vector_right_type cvr2 = Kokkos::subview( mv , Kokkos::ALL() , 1 );
    const_vector_right_type cvr3 = Kokkos::subview( mv , Kokkos::ALL() , 2 );

    ASSERT_TRUE( & v1[0] == & v1(0) );
    ASSERT_TRUE( & v1[0] == & mv(0,0) );
    ASSERT_TRUE( & v2[0] == & mv(0,1) );
    ASSERT_TRUE( & v3[0] == & mv(0,2) );

    ASSERT_TRUE( & cv1[0] == & mv(0,0) );
    ASSERT_TRUE( & cv2[0] == & mv(0,1) );
    ASSERT_TRUE( & cv3[0] == & mv(0,2) );

    ASSERT_TRUE( & vr1[0] == & mv(0,0) );
    ASSERT_TRUE( & vr2[0] == & mv(0,1) );
    ASSERT_TRUE( & vr3[0] == & mv(0,2) );

    ASSERT_TRUE( & cvr1[0] == & mv(0,0) );
    ASSERT_TRUE( & cvr2[0] == & mv(0,1) );
    ASSERT_TRUE( & cvr3[0] == & mv(0,2) );

    ASSERT_TRUE( & mv1(0,0) == & mv( 1 , 2 ) );
    ASSERT_TRUE( & mv1(1,1) == & mv( 2 , 3 ) );
    ASSERT_TRUE( & mv1(3,2) == & mv( 4 , 4 ) );
    ASSERT_TRUE( & mvr1(0,0) == & mv_right( 1 , 2 ) );
    ASSERT_TRUE( & mvr1(1,1) == & mv_right( 2 , 3 ) );
    ASSERT_TRUE( & mvr1(3,2) == & mv_right( 4 , 4 ) );

    const_vector_type c_cv1( v1 );
    typename vector_type::const_type c_cv2( v2 );
    typename const_vector_type::const_type c_ccv2( v2 );

    const_multivector_type cmv( mv );
    typename multivector_type::const_type cmvX( cmv );
    typename const_multivector_type::const_type ccmvX( cmv );
*/
  }
};

} // namespace Test

/*--------------------------------------------------------------------------*/

