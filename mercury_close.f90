! Validated reader for Chambers' MERCURY6 encounter format. CPU postprocessing only.
module mercury_close
  use iso_fortran_env, only: int64, iostat_end, iostat_eor
  use, intrinsic :: ieee_arithmetic
  use mercury_support, only: fail, read_messages, lower, next_line
  implicit none
  private
  public :: close_run, close_elements, close_field
  real(8), parameter :: k2=2.959122082855911d-4, pi=3.1415926535897932384626433832795d0
  integer, parameter :: maxcode=224**3-1, batch_size=256
  type object
    character(25) :: name=''
    logical :: wanted=.false.
    integer :: output=0, stamp=0
  end type
  type(object), allocatable :: objects(:)
  integer, allocatable :: table(:), mapping(:)
  real(8), allocatable :: masses(:)
  integer :: used=0, stamp=0, style=0, record_number=0
  character(250) :: current_file='', header=''
  integer :: header_length
  logical :: select_all
contains
  subroutine bad(reason)
    character(*), intent(in) :: reason
    character(32) :: location
    write(location,'(i0)') record_number
    call fail(trim(current_file)//': record '//trim(location)//': '//reason)
  end subroutine

  integer function hash(name,n) result(slot)
    character(*), intent(in) :: name
    integer, intent(in) :: n
    integer(int64) :: h
    integer :: j
    h=0
    do j=1,len_trim(name)
      h=mod(h*131+iachar(name(j:j)),int(n,int64))
    end do
    slot=int(h)+1
  end function

  subroutine grow_objects()
    type(object), allocatable :: replacement(:)
    integer :: j,s,n
    n=256
    if(allocated(objects)) n=2*size(objects)
    allocate(replacement(n))
    if(allocated(objects)) replacement(:used)=objects(:used)
    call move_alloc(replacement,objects)
    if(allocated(table)) deallocate(table)
    allocate(table(2*n+1)); table=0
    do j=1,used
      s=hash(objects(j)%name,size(table))
      do while(table(s)/=0)
        s=mod(s,size(table))+1
      end do
      table(s)=j
    end do
  end subroutine

  integer function intern(name) result(index_)
    character(*), intent(in) :: name
    integer :: s
    if(.not.allocated(objects)) call grow_objects()
    if(used==size(objects)) call grow_objects()
    s=hash(name,size(table))
    do while(table(s)/=0)
      index_=table(s)
      if(objects(index_)%name==name) return
      s=mod(s,size(table))+1
    end do
    used=used+1; index_=used
    objects(index_)%name=name
    table(s)=index_
  end function

  subroutine ensure_code(code)
    integer, intent(in) :: code
    integer, allocatable :: replacement(:)
    real(8), allocatable :: new_masses(:)
    integer :: n,old
    if(code<1.or.code>maxcode) call bad('body code out of range')
    old=0
    if(allocated(mapping)) old=size(mapping)
    if(code<=old) return
    n=min(maxcode,max(code,max(256,2*old)))
    allocate(replacement(n),new_masses(n)); replacement=0; new_masses=0d0
    if(old>0) then
      replacement(:old)=mapping; new_masses(:old)=masses
    end if
    call move_alloc(replacement,mapping); call move_alloc(new_masses,masses)
  end subroutine

  subroutine encoded(bytes)
    character(*), intent(in) :: bytes
    integer :: j
    do j=1,len(bytes)
      if(iachar(bytes(j:j))<32) call bad('invalid base-224 byte')
    end do
  end subroutine

  real(8) function fraction(bytes) result(x)
    character(*), intent(in) :: bytes
    integer :: j
    call encoded(bytes)
    x=0d0
    do j=len(bytes),1,-1
      x=(x+real(iachar(bytes(j:j))-32,8))/224d0
    end do
  end function

  real(8) function floating(bytes) result(x)
    character(8), intent(in) :: bytes
    call encoded(bytes)
    x=(2d0*fraction(bytes(:7))-1d0)*10d0**(iachar(bytes(8:8))-144)
    if(.not.ieee_is_finite(x)) call bad('nonfinite number')
  end function

  integer function code_value(bytes) result(code)
    character(3), intent(in) :: bytes
    call encoded(bytes)
    code=(iachar(bytes(1:1))-32)*224**2+(iachar(bytes(2:2))-32)*224+iachar(bytes(3:3))-32
  end function

  subroutine record(unit,line,length,eof)
    integer, intent(in) :: unit
    character(*), intent(out) :: line
    integer, intent(out) :: length
    logical, intent(out) :: eof
    integer :: ios
    read(unit,'(a)',advance='no',size=length,iostat=ios) line
    eof=ios==iostat_end.and.length==0
    if(eof) return
    record_number=record_number+1
    if(ios/=iostat_eor.and.ios/=iostat_end) call bad('record too long or unreadable')
    if(length>0) then
      if(iachar(line(length:length))==13) length=length-1
    end if
  end subroutine

  subroutine config(files,nfiles)
    character(250), intent(out) :: files(50)
    integer, intent(out) :: nfiles
    character(1024) :: line,value
    integer :: unit,ios,j,k,idx
    current_file='close.in'; record_number=0
    open(newunit=unit,file='close.in',status='old',iostat=ios)
    if(ios/=0) call bad('cannot open file')
    call setting(unit,line)
    k=index(line,'=',back=.true.)
    read(line(k+1:),*,iostat=ios) nfiles
    if(ios/=0) call bad('invalid input file count')
    if(nfiles<1.or.nfiles>50) call bad('input file count must be 1 to 50')
    do j=1,nfiles
      call setting(unit,line)
      if(len_trim(line)>250) call bad('input filename too long')
      files(j)=trim(line)
    end do
    call setting(unit,line)
    k=index(line,'=',back=.true.); value=trim(lower(adjustl(line(k+1:))))
    select case(trim(value))
    case('days'); style=0
    case('years'); style=1
    case default; call bad('time units must be days or years')
    end select
    call setting(unit,line)
    k=index(line,'=',back=.true.); value=trim(lower(adjustl(line(k+1:))))
    select case(trim(value))
    case('yes'); style=style+2
    case('no')
    case default; call bad('relative time must be yes or no')
    end select
    select_all=.true.
    do
      call next_line(unit,line,ios)
      if(ios==iostat_end) exit
      record_number=record_number+1
      if(ios/=0) call bad('cannot read selection')
      call valid_name(trim(line))
      idx=intern(trim(line)); objects(idx)%wanted=.true.; select_all=.false.
    end do
    close(unit)
    call m_formce(style,line,header,header_length)
  end subroutine

  subroutine setting(unit,line)
    integer, intent(in) :: unit
    character(*), intent(out) :: line
    integer :: ios
    call next_line(unit,line,ios)
    record_number=record_number+1
    if(ios/=0) call bad('missing setting')
  end subroutine

  subroutine valid_name(name)
    character(*), intent(in) :: name
    integer :: j
    if(len_trim(name)<1.or.len_trim(name)>25) call bad('body name must contain 1 to 25 characters')
    do j=1,len_trim(name)
      if(iachar(name(j:j))<=32.or.iachar(name(j:j))>126) call bad('invalid body name')
    end do
  end subroutine

  function basename(name) result(out)
    character(25), intent(in) :: name
    character(25) :: out
    integer :: j
    out=name
    do j=1,len_trim(out)
      if(index('*/.:&'//achar(92),out(j:j))>0) out(j:j)='_'
    end do
  end function

  subroutine filenames()
    integer, allocatable :: slots(:)
    integer :: j,s,k
    character(25) :: name
    allocate(slots(2*used+1)); slots=0
    do j=1,used
      if(.not.objects(j)%wanted) cycle
      name=basename(objects(j)%name); s=hash(name,size(slots))
      do while(slots(s)/=0)
        k=slots(s)
        if(basename(objects(k)%name)==name) call bad('output filename collision: '//trim(name)//'.clo')
        s=mod(s,size(slots))+1
      end do
      slots(s)=j
    end do
  end subroutine

  subroutine scan(filename,write_output)
    character(*), intent(in) :: filename
    logical, intent(in) :: write_output
    integer :: unit,ios,length,nbig,nsml,j,code,idx,algorithm,precision,codes(2),indices(2),p
    real(8) :: central,rcen,rmax,origin,time,distance,x(3),v(3),elements(3,2),unused
    character(1024) :: line
    logical :: eof,have_header
    current_file=filename; record_number=0; have_header=.false.
    open(newunit=unit,file=filename,status='old',iostat=ios)
    if(ios/=0) call bad('cannot open file')
    do
      call record(unit,line,length,eof)
      if(eof) exit
      if(length<3) call bad('truncated record prefix')
      if(line(:2)/=achar(12)//'6') call bad('expected MERCURY6 encounter record')
      select case(line(3:3))
      case('a')
        if(length/=68) call bad('header must contain 68 bytes')
        read(line(4:5),'(i2)',iostat=ios) algorithm
        if(ios/=0) call bad('invalid algorithm')
        if(.not.any(algorithm==[1,2,3,4,9,10,11,12])) call bad('invalid algorithm')
        read(line(68:68),'(i1)',iostat=ios) precision
        if(ios/=0) call bad('invalid precision')
        if(precision<1.or.precision>3) call bad('invalid precision')
        time=floating(line(6:13)); nbig=code_value(line(14:16)); nsml=code_value(line(17:19))
        if(int(nbig,int64)+nsml>maxcode) call bad('body count exceeds encounter encoding capacity')
        central=floating(line(20:27))*k2
        do j=28,44,8
          unused=floating(line(j:j+7))
        end do
        rcen=floating(line(52:59)); rmax=floating(line(60:67))
        if(central<=0d0.or.rcen<=0d0.or.rmax<=rcen) call bad('invalid central mass or radial range')
        if(.not.have_header) origin=time
        have_header=.true.
        if(allocated(mapping)) mapping=0
        stamp=stamp+1
        do j=1,nbig+nsml
          call record(unit,line,length,eof)
          if(eof.or.length/=68) call bad('truncated body metadata: expected 68 bytes')
          code=code_value(line(:3)); call ensure_code(code)
          if(mapping(code)/=0) call bad('duplicate body code')
          call valid_name(trim(line(4:28)))
          idx=intern(line(4:28))
          if(objects(idx)%stamp==stamp) call bad('duplicate body name')
          objects(idx)%stamp=stamp; mapping(code)=idx
          masses(code)=floating(line(29:36))*k2
          if(masses(code)<0d0) call bad('negative body mass')
          do p=37,61,8
            unused=floating(line(p:p+7))
          end do
          if(unused<0d0) call bad('negative density')
          if(select_all) objects(idx)%wanted=.true.
        end do
      case('b')
        if(.not.have_header) call bad('encounter before metadata header')
        if(length/=73) call bad('encounter must contain 73 bytes')
        time=floating(line(4:11)); codes(1)=code_value(line(12:14)); codes(2)=code_value(line(15:17))
        if(codes(1)==codes(2)) call bad('encounter must involve different bodies')
        do j=1,2
          code=codes(j)
          if(code<1) call bad('invalid encounter body code')
          if(.not.allocated(mapping)) call bad('encounter has no active bodies')
          if(code>size(mapping)) call bad('unmapped encounter body code')
          if(mapping(code)==0) call bad('unmapped encounter body code')
          indices(j)=mapping(code)
          p=26+(j-1)*24
          call state(line(p:p+23),rcen,rmax,central,x,v)
          call close_elements(central+masses(code),x,v,elements(1,j),elements(2,j),elements(3,j))
        end do
        distance=floating(line(18:25))
        if(distance<0d0) call bad('negative encounter distance')
        ! Check date range even for an unselected encounter, before opening outputs.
        if(style==1.and.abs(time)>1d12) call bad('calendar time is outside the supported range (+/- 1e12 JD)')
        if(write_output) then
          call row(objects(indices(1))%output,objects(indices(2))%name,time,origin,distance,elements(:,1),elements(:,2))
          call row(objects(indices(2))%output,objects(indices(1))%name,time,origin,distance,elements(:,2),elements(:,1))
        end if
      case default
        call bad('unknown encounter record type')
      end select
    end do
    close(unit)
    if(.not.have_header) call bad('missing metadata header')
  end subroutine

  subroutine state(bytes,rcen,rmax,central,x,v)
    character(24), intent(in) :: bytes
    real(8), intent(in) :: rcen,rmax,central
    real(8), intent(out) :: x(3),v(3)
    real(8) :: r,theta,phi,fv,vt,vp,speed
    r=rcen*10d0**(fraction(bytes(:4))*log10(rmax/rcen))
    theta=pi*fraction(bytes(5:8)); phi=2*pi*fraction(bytes(9:12))
    fv=fraction(bytes(13:16)); vt=pi*fraction(bytes(17:20)); vp=2*pi*fraction(bytes(21:24))
    if(fv<=0d0) call bad('zero velocity encoding fraction')
    ! mio_ce calls mco_x2ov with m=0: velocity compression uses central GM only.
    speed=sqrt(2d0*sqrt(.5d0*(1d0/fv-1d0))*central/r)
    x=r*[sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)]
    v=speed*[sin(vt)*cos(vp),sin(vt)*sin(vp),cos(vt)]
  end subroutine

  subroutine close_elements(mu,x,v,a,e,inclination)
    real(8), intent(in) :: mu,x(3),v(3)
    real(8), intent(out) :: a,e,inclination
    real(8) :: r,scale,u(3),rh(3),h(3),ev(3),alpha,v2,angular_error(3)
    if(.not.ieee_is_finite(mu).or.any(.not.ieee_is_finite(x)).or.any(.not.ieee_is_finite(v))) &
      call bad('nonfinite encounter state')
    scale=maxval(abs(x))
    if(mu<=0d0.or.scale<=0d0) call bad('invalid encounter state')
    r=scale*sqrt(sum((x/scale)**2)); rh=x/r
    u=v/sqrt(mu/r)
    h=[rh(2)*u(3)-rh(3)*u(2),rh(3)*u(1)-rh(1)*u(3),rh(1)*u(2)-rh(2)*u(1)]
    ev=(dot_product(u,u)-1d0)*rh-dot_product(rh,u)*u
    e=sqrt(dot_product(ev,ev))
    v2=dot_product(v,v)
    alpha=2d0/r-v2/mu
    if(alpha==0d0) then
      a=ieee_value(0d0,ieee_positive_inf)
    else
      a=1d0/alpha
    end if
    ! A radial state at an arbitrary angle can leave cancellation-sized h.
    ! Component-wise product bounds retain resolved, very small angular momenta.
    angular_error=16d0*epsilon(1d0)*[abs(rh(2)*u(3))+abs(rh(3)*u(2)), &
      abs(rh(3)*u(1))+abs(rh(1)*u(3)),abs(rh(1)*u(2))+abs(rh(2)*u(1))]
    if(all(abs(h)<=angular_error)) then
      inclination=ieee_value(0d0,ieee_quiet_nan)
    else
      inclination=atan2(sqrt(h(1)**2+h(2)**2),h(3))*180d0/pi
    end if
  end subroutine

  function close_field(value,format) result(text)
    real(8), intent(in) :: value
    character(*), intent(in) :: format
    character(:), allocatable :: text
    character(64) :: buffer
    integer :: width,ios
    if(ieee_is_nan(value)) then
      text='NaN'; return
    end if
    if(.not.ieee_is_finite(value)) then
      text='Infinity'; return
    end if
    read(format(3:index(format,'.')-1),*,iostat=ios) width
    if(ios/=0) call fail('Invalid internal close6 format')
    write(buffer,format) value
    text=buffer(:width)
    if(index(text,'*')>0) then
      write(buffer,'(es17.8e3)') value
      text=buffer(:17)
    end if
  end function

  subroutine calendar(jd,year,month,day)
    real(8), intent(in) :: jd
    integer(int64), intent(out) :: year
    integer, intent(out) :: month
    real(8), intent(out) :: day
    integer(int64) :: z,a,b,c,d,ee,g,alpha
    real(8) :: f
    ! Duffett-Smith / Chambers calendar convention: Julian before 1582-10-15,
    ! Gregorian thereafter. FLOOR and 64-bit integers also cover negative JD.
    z=floor(jd+.5d0,kind=int64); f=jd+.5d0-real(z,8)
    a=z
    if(z>=2299161_int64) then
      alpha=floor((real(z,8)-1867216.25d0)/36524.25d0,kind=int64)
      a=z+1+alpha-floor(real(alpha,8)/4d0,kind=int64)
    end if
    b=a+1524
    c=floor((real(b,8)-122.1d0)/365.25d0,kind=int64)
    d=floor(365.25d0*real(c,8),kind=int64)
    ee=floor(real(b-d,8)/30.6001d0,kind=int64)
    g=floor(30.6001d0*real(ee,8),kind=int64)
    day=real(b-d-g,8)+f
    month=int(ee-1)
    if(ee>13) month=int(ee-13)
    year=c-4716
    if(month<=2) year=c-4715
  end subroutine

  subroutine row(unit,name,time,origin,distance,own,other)
    integer, intent(in) :: unit
    character(25), intent(in) :: name
    real(8), intent(in) :: time,origin,distance,own(3),other(3)
    character(:), allocatable :: line
    character(32) :: year_text,month_text
    integer(int64) :: year
    integer :: month,ios
    real(8) :: clock,day
    if(unit==0) return
    clock=time
    select case(style)
    case(1)
      call calendar(time,year,month,day)
      write(year_text,'(i10)') year
      if(index(year_text,'*')>0) write(year_text,'(i0)') year
      write(month_text,'(i2)') month
      line=' '//trim(year_text)//' '//trim(month_text)//' '//close_field(day,'(f8.5)')
    case(2)
      line=' '//close_field(time-origin,'(f18.5)')
    case(3)
      line=' '//close_field((time-origin)/365.25d0,'(f18.7)')
    case default
      line=' '//close_field(clock,'(f18.5)')
    end select
    line=line//' '//name//' '//close_field(distance,'(f10.8)') &
      //' '//close_field(own(1),'(f9.4)')//' '//close_field(own(2),'(f8.6)')//' '//close_field(own(3),'(f7.3)') &
      //' '//close_field(other(1),'(f9.4)')//' '//close_field(other(2),'(f8.6)')//' '//close_field(other(3),'(f7.3)')
    write(unit,'(a)',iostat=ios) line
    if(ios/=0) call bad('cannot write close encounter output')
  end subroutine

  subroutine close_run()
    character(250) :: files(50)
    character(80) :: messages(200)
    integer :: lengths(200),nfiles,j,k,start_,finish_,unit,ios,nselected,p
    integer, allocatable :: selected(:)
    logical :: exists
    if(allocated(objects)) deallocate(objects,table)
    if(allocated(mapping)) deallocate(mapping,masses)
    used=0; stamp=0
    current_file='message.in'; record_number=0
    open(newunit=unit,file='message.in',status='old',iostat=ios)
    if(ios/=0) call bad('cannot open file')
    call read_messages(unit,lengths,messages)
    close(unit)
    call config(files,nfiles)
    ! Preflight all records before creating even the first output file.
    do j=1,nfiles
      call scan(files(j),.false.)
    end do
    if(used==0) return ! A valid empty population needs no output files.
    call filenames()
    selected=pack([(j,j=1,used)],objects(:used)%wanted)
    nselected=size(selected)
    start_=1
    do while(start_<=nselected)
      finish_=min(nselected,start_+batch_size-1)
      do p=start_,finish_
        k=selected(p)
        inquire(file=trim(basename(objects(k)%name))//'.clo',exist=exists)
        if(exists) then
          write(*,'(a)') 'WARNING: existing output skipped: '//trim(basename(objects(k)%name))//'.clo'
          cycle
        end if
        open(newunit=unit,file=trim(basename(objects(k)%name))//'.clo',status='new',iostat=ios)
        if(ios/=0) call bad('cannot create output for '//trim(objects(k)%name))
        objects(k)%output=unit
        write(unit,'(/,30x,a25,//,a)',iostat=ios) objects(k)%name,header(:header_length)
        if(ios/=0) call bad('cannot write output header')
      end do
      do j=1,nfiles
        call scan(files(j),.true.)
      end do
      do p=start_,finish_
        k=selected(p)
        if(objects(k)%output/=0) close(objects(k)%output)
        objects(k)%output=0
      end do
      start_=finish_+1
    end do
  end subroutine
end module
