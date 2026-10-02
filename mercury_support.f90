! Runtime support for the original fixed-form Mercury drivers.
module mercury_support
  use iso_fortran_env, only: int64, error_unit
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  integer :: backend_request = 0 ! 0 CPU, 1 CUDA, 2 automatic
  integer :: force_model_version = 0
  integer(int64) :: accepted_steps = 0, rejected_steps = 0, force_calls = 0
  real(8) :: initial_step = 0d0, step_seconds = 0d0, event_seconds = 0d0
  character(25), allocatable :: names(:)
contains
  real(8) function wall_seconds()
    integer(int64) :: ticks,rate
    call system_clock(ticks,rate)
    wall_seconds=real(ticks,8)/real(rate,8)
  end function
  subroutine fail(message)
    character(*), intent(in) :: message
    write(error_unit,'(a)') 'MERCURY: '//trim(message)
    error stop 1
  end subroutine

  subroutine validate_separations(n,nbig,x,epochs)
    integer, intent(in) :: n,nbig
    real(8), intent(in) :: x(3,n)
    real(8), intent(in), optional :: epochs(n)
    integer :: i,j
    do i=2,nbig
      do j=i+1,n
        if(present(epochs)) then
          if(epochs(i)/=epochs(j)) cycle
        endif
        if(all(x(:,i)==x(:,j))) call fail('Nonfinite gravity: coincident interacting bodies')
      enddo
    enddo
  end subroutine

  subroutine read_messages(unit,lengths,messages)
    integer, intent(in) :: unit
    integer, intent(out) :: lengths(:)
    character(*), intent(out) :: messages(:)
    integer :: index_,length_,ios
    character(80) :: message
    logical :: seen(size(lengths))
    seen=.false.; lengths=0; messages=''
    do
      read(unit,'(i3,1x,i2,1x,a80)',iostat=ios) index_,length_,message
      if(ios<0) exit
      if(ios/=0) call fail('Invalid message.in record')
      if(index_<1.or.index_>size(lengths)) call fail('Invalid message.in index')
      if(length_<0.or.length_>len(messages(1))) call fail('Invalid message.in length')
      if(seen(index_)) call fail('Duplicate message.in index')
      seen(index_)=.true.; lengths(index_)=length_; messages(index_)=message
    end do
    if(.not.all(seen)) call fail('Incomplete message.in')
  end subroutine

  function lower(text) result(out)
    character(*), intent(in) :: text
    character(len(text)) :: out
    integer :: i,k
    out=text
    do i=1,len(text)
      k=iachar(text(i:i))
      if(k>=65.and.k<=90) out(i:i)=achar(k+32)
    end do
  end function

  subroutine next_line(unit,line,ios)
    integer, intent(in) :: unit
    character(*), intent(out) :: line
    integer, intent(out) :: ios
    do
      read(unit,'(a)',iostat=ios) line
      if(ios/=0) return
      line=adjustl(line)
      if(len_trim(line)==0) cycle
      if(line(1:1)==')') cycle
      return
    end do
  end subroutine

  subroutine read_extras(unit)
    integer, intent(in) :: unit
    integer :: ios,k
    character(1024) :: line,key,value
    do
      call next_line(unit,line,ios)
      if(ios<0) exit
      if(ios/=0) call fail('Cannot read optional integration settings')
      k=index(line,'=')
      if(k==0) call fail('Expected key = value in optional settings')
      key=trim(lower(adjustl(line(:k-1))))
      value=trim(lower(adjustl(line(k+1:))))
      select case(trim(key))
      case('execution backend')
        call set_backend(value)
      case('force model version')
        read(value,*,iostat=ios) force_model_version
        if(ios/=0.or.(force_model_version/=1.and.force_model_version/=2)) call fail('Unsupported force model version')
      case default
        call fail('Unknown optional setting: '//trim(key))
      end select
    end do
  end subroutine

  subroutine set_backend(value)
    character(*), intent(in) :: value
    select case(trim(lower(value)))
    case('cpu');  backend_request=0
    case('cuda'); backend_request=1
    case('auto'); backend_request=2
    case default; call fail('execution backend must be cpu, cuda, or auto')
    end select
  end subroutine

  subroutine backend_override(filename)
    ! As with Mercury's other restart settings, dynamics come from the dump.
    ! Only the execution device may be overridden by the ordinary input file.
    character(*), intent(in) :: filename
    integer :: unit,ios,k
    character(1024) :: line
    open(newunit=unit,file=filename,status='old',iostat=ios)
    if(ios/=0) return
    do
      call next_line(unit,line,ios)
      if(ios/=0) exit
      k=index(line,'=')
      if(k==0) cycle
      if(trim(lower(line(:k-1)))=='execution backend') call set_backend(trim(adjustl(line(k+1:))))
    end do
    close(unit)
  end subroutine

  subroutine write_extras(unit)
    integer, intent(in) :: unit
    character(4), parameter :: backends(0:2) = ['cpu ','cuda','auto']
    write(unit,'(a)') ' execution backend = '//trim(backends(backend_request))
    write(unit,'(a)') ' force model version = 2'
  end subroutine

  subroutine register_name(name)
    ! Open addressing replaces an O(N**2) duplicate-name scan at startup.
    character(*), intent(in) :: name
    integer(int64) :: h
    integer :: i,slot,n
    if(.not.allocated(names)) call fail('Name table was not initialized')
    n=size(names)
    h=0
    do i=1,len_trim(name)
      h=mod(h*131+iachar(name(i:i)),int(n,int64))
    end do
    slot=int(h)+1
    do i=1,n
      if(names(slot)==name) call fail('Duplicate body name: '//trim(name))
      if(names(slot)=='') then
        names(slot)=name
        return
      end if
      slot=mod(slot,n)+1
    end do
    call fail('Body name table is full')
  end subroutine

  real(8) function epoch_epsilon(t,target)
    real(8), intent(in) :: t,target
    epoch_epsilon=max(1d-13,2d0*spacing(max(abs(t),abs(target))))
  end function

  real(8) function clipped_step(t,target,proposed,cap)
    real(8), intent(in) :: t,target,proposed,cap
    clipped_step=sign(min(abs(target-t),abs(proposed),abs(cap)),target-t)
    if(.not.ieee_is_finite(clipped_step).or.t+clipped_step==t) &
      call fail('Timestep cannot advance the epoch; loosen tolerance or inspect the orbit')
  end function

  integer function count_bodies(filename,big) result(n)
    character(*), intent(in) :: filename
    logical, intent(in) :: big
    integer :: unit,ios,i
    character(1024) :: line
    real(8) :: state(9)
    open(newunit=unit,file=filename,status='old',iostat=ios)
    if(ios/=0) call fail('Cannot open '//trim(filename))
    call next_line(unit,line,ios) ! coordinate style
    if(ios/=0) call fail('Missing coordinate style in '//trim(filename))
    if(big) call next_line(unit,line,ios) ! big-body epoch
    n=0
    do
      call next_line(unit,line,ios)
      if(ios<0) exit
      if(ios/=0) call fail('Cannot read '//trim(filename))
      n=n+1
      call next_line(unit,line,ios)
      if(ios/=0) call fail('Missing coordinates in '//trim(filename))
      backspace(unit)
      read(unit,*,iostat=ios) state
      if(ios/=0) call fail('Expected nine coordinates/elements and spins in '//trim(filename))
      if(any(.not.ieee_is_finite(state))) call fail('Nonfinite input state in '//trim(filename))
    end do
    close(unit)
  end function
end module

subroutine mercury_capacity(kind)
  use mercury_support
  implicit none
  character(*), intent(in) :: kind
  integer :: nmax,cmax
  common /mercury_sizes/ nmax,cmax
  save /mercury_sizes/
  integer :: unit,ios,i,j,nb,ns,nfiles,other,names_count
  integer(int64) :: pairs
  character(1024) :: line,paths(10)
  character(80) :: header
  logical :: restart
  nmax=1
  cmax=5000
  if(kind=='mercury') then
    open(newunit=unit,file='files.in',status='old',iostat=ios)
    if(ios/=0) call fail('Cannot open files.in')
    do i=1,10
      call next_line(unit,paths(i),ios)
      if(ios/=0) call fail('files.in must contain ten filenames')
    end do
    close(unit)
    inquire(file=trim(paths(10)),exist=restart)
    j=0
    if(restart) j=6
    nb=count_bodies(trim(paths(j+1)),.true.)
    ns=count_bodies(trim(paths(j+2)),.false.)
    nmax=nb+ns+1
    ! One minimum/collision per interacting pair per accepted step, plus
    ! central impacts. Allocate from input size instead of a fixed limit.
    pairs=int(nb,int64)*ns+int(nb,int64)*(nb-1)/2
    if(pairs>huge(cmax)-nmax) call fail('Encounter capacity exceeds integer range')
    cmax=max(5000,nmax,int(pairs))
    allocate(names(2*nmax+1))
    names=''
  else
    open(newunit=unit,file=trim(kind)//'.in',status='old',iostat=ios)
    if(ios/=0) call fail('Cannot open '//trim(kind)//'.in')
    call next_line(unit,line,ios)
    j=index(line,'=',back=.true.)
    read(line(j+1:),*,iostat=ios) nfiles
    if(ios/=0.or.nfiles<1.or.nfiles>50) call fail('Invalid post-processor file count')
    do i=1,nfiles
      call next_line(unit,line,ios)
      if(ios/=0) call fail('Missing post-processor input filename')
      open(newunit=other,file=trim(line),status='old',iostat=ios)
      if(ios/=0) call fail('Cannot open '//trim(line))
      read(other,'(a)',iostat=ios) header
      close(other)
      if(ios/=0.or.header(1:3)/=achar(12)//'6a') call fail('Expected Mercury6 output header')
      if(len_trim(header)<19) call fail('Truncated Mercury6 output header')
      nb=0; ns=0
      do j=1,3
        if(iachar(header(13+j:13+j))<32.or.iachar(header(16+j:16+j))<32) &
          call fail('Invalid Mercury6 body-count encoding')
        nb=224*nb+iachar(header(13+j:13+j))-32
        ns=224*ns+iachar(header(16+j:16+j))-32
      end do
      nmax=nmax+nb+ns
    end do
    ! Explicit selection lists can be longer than the bodies in one file.
    names_count=0
    do
      call next_line(unit,line,ios)
      if(ios/=0) exit
      names_count=names_count+1
    end do
    nmax=nmax+names_count
    close(unit)
  end if
  if(nmax>11239424) call fail('Body count exceeds the Mercury6 output encoding')
end subroutine
