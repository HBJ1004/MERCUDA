! Narrow C interoperability layer. The ordinary MERCURY driver owns I/O/events.
module mercury_gpu
  use iso_c_binding
  use mercury_support, only: backend_request, fail, force_calls, rejected_steps
  implicit none
  logical :: gpu_enabled=.false., gpu_dirty=.true., host_current=.true.
  integer :: current_n=0, event_size=0
  type, bind(C) :: gpu_event
    integer(c_int) :: i,j,kind,now
    real(c_double) :: distance,time,xi(6),xj(6)
  end type
  type(gpu_event), pointer :: events(:)=>null()
  interface
    integer(c_int) function update_limits(rce) bind(C,name="mercury_cuda_limits")
      import
      real(c_double),intent(in) :: rce(*)
    end function
    integer(c_int) function hybrid_begin(h,crit,flag,cap,ce,nce,pi,pj,x,v,nf) bind(C,name="mercury_cuda_hybrid_begin")
      import
      real(c_double),value :: h
      integer(c_int),value :: flag,cap
      real(c_double) :: crit(*),x(*),v(*)
      integer(c_int) :: ce(*),nce,pi(*),pj(*)
      integer(c_int64_t) :: nf
    end function
    integer(c_int) function hybrid_finish(h,m,x,v,nf) bind(C,name="mercury_cuda_hybrid_finish")
      import
      real(c_double),value :: h
      real(c_double) :: m(*),x(*),v(*)
      integer(c_int64_t) :: nf
    end function
    integer(c_int) function encounter_enter(n,nb,m,x,v,crit,rce,rphys,np,pi,pj) bind(C,name="mercury_cuda_encounter_enter")
      import
      integer(c_int),value :: n,nb,np
      real(c_double) :: m(*),x(*),v(*),crit(*),rce(*),rphys(*)
      integer(c_int) :: pi(*),pj(*)
    end function
    integer(c_int) function encounter_update(m,x,v) bind(C,name="mercury_cuda_encounter_update")
      import
      real(c_double) :: m(*),x(*),v(*)
    end function
    subroutine encounter_exit() bind(C,name="mercury_cuda_encounter_exit")
    end subroutine
    integer(c_int) function configure(algorithm) bind(C,name="mercury_cuda_configure")
      import
      integer(c_int),value :: algorithm
    end function
    integer(c_int) function available() bind(C,name='mercury_cuda_available')
      import
    end function
    integer(c_int) function upload(n,nbig,pn,ngflag,m,x,v,ngf,jcen,rce,rphys) bind(C,name='mercury_cuda_upload')
      import
      integer(c_int),value :: n,nbig,pn,ngflag
      real(c_double),intent(in) :: m(*),x(*),v(*),ngf(*),jcen(*),rce(*),rphys(*)
    end function
    integer(c_int) function step(time,h,hdid,tol,forces,rejected) bind(C,name='mercury_cuda_step')
      import
      real(c_double),value :: time,tol
      real(c_double) :: h,hdid
      integer(c_int64_t) :: forces,rejected
    end function
    integer(c_int) function download(x,v,previous) bind(C,name='mercury_cuda_download')
      import
      real(c_double) :: x(*),v(*)
      integer(c_int),value :: previous
    end function
    integer(c_int) function find_events(time,h,rcen,ptr,count) bind(C,name='mercury_cuda_events')
      import
      real(c_double),value :: time,h,rcen
      type(c_ptr) :: ptr
      integer(c_int) :: count
    end function
    integer(c_int) function export_state(h,physical,x,v) bind(C,name="mercury_cuda_export")
      import
      real(c_double),value :: h
      integer(c_int),value :: physical
      real(c_double) :: x(*),v(*)
    end function
    subroutine reset_history(flag) bind(C,name="mercury_cuda_reset")
      import
      integer(c_int),value :: flag
    end subroutine
    subroutine gpu_free() bind(C,name='mercury_cuda_free')
    end subroutine
  end interface
contains
  subroutine gpu_limits(rce)
    real(8),intent(in) :: rce(*)
    if(.not.gpu_enabled.or.gpu_dirty) return
    if(update_limits(rce)/=0) call fail('CUDA encounter-radius update failed')
  end subroutine
  subroutine gpu_hybrid_begin(h,crit,flag,cap,ce,nce,pi,pj,x,v)
    real(8) :: h,crit(*),x(3,*),v(3,*)
    integer :: flag,cap,ce(*),nce,pi(*),pj(*)
    integer(c_int64_t) :: nf
    if(hybrid_begin(h,crit,flag,cap,ce,nce,pi,pj,x,v,nf)/=0) call fail('CUDA hybrid drift failed')
    force_calls=force_calls+nf
    flag=2
  end subroutine
  subroutine gpu_hybrid_finish(h,m,x,v)
    real(8) :: h,m(*),x(3,*),v(3,*)
    integer(c_int64_t) :: nf
    if(hybrid_finish(h,m,x,v,nf)/=0) call fail('CUDA hybrid kick failed')
    force_calls=force_calls+nf
    host_current=.false.
  end subroutine
  subroutine gpu_refresh_events(time,h,rcen)
    real(8) :: time,h,rcen
    type(c_ptr) :: ptr
    if(find_events(time,h,rcen,ptr,event_size)/=0) call fail('CUDA event refresh failed')
    nullify(events)
    if(event_size>0) call c_f_pointer(ptr,events,[event_size])
  end subroutine
  subroutine gpu_select(algor,n,nbig,m,opt,unit)
    integer,intent(in) :: algor,n,nbig,opt(8),unit
    real(8),intent(in) :: m(n)
    logical :: supported
    supported=(algor==1.or.algor==2.or.algor==3.or.algor==4.or.algor==9.or.algor==10).and.opt(8)==0
    if(backend_request==1.and..not.supported) &
      call fail('CUDA requires MVS/BS/BS2/RADAU/HYBRID and user-defined force = no')
    if(backend_request==1.and.available()==0) call fail('CUDA was requested but no CUDA device/build is available')
    gpu_enabled=backend_request/=0.and.supported.and.available()/=0
    if(backend_request==2.and.n-nbig<4096) gpu_enabled=.false.
    gpu_dirty=.true.; host_current=.true.; current_n=0
    if(gpu_enabled) then
      if(configure(algor)/=0) call fail('CUDA algorithm initialization failed')
      write(unit,'(a,i2)') ' Execution backend: CUDA, algorithm ',algor
    else
      write(unit,'(a)') ' Execution backend: CPU'
    endif
  end subroutine
  subroutine gpu_push(n,nbig,m,x,v,ngf,jcen,rce,rphys,opt,ngflag)
    integer,intent(in) :: n,nbig,opt(8),ngflag
    real(8),intent(in) :: m(n),x(3,n),v(3,n),ngf(4,n),jcen(3),rce(n),rphys(n)
    if(.not.gpu_enabled.or..not.gpu_dirty) return
    if(upload(n,nbig,opt(7),ngflag,m,x,v,ngf,jcen,rce,rphys)/=0) then
      if(backend_request==2.and.current_n==0) then
        write(*,'(a)') ' CUDA initialization failed; auto selected CPU.'
        gpu_enabled=.false.; call gpu_free(); return
      endif
      call fail('CUDA state upload failed')
    endif
    current_n=n; gpu_dirty=.false.; host_current=.true.
  end subroutine
  subroutine gpu_advance(time,h,hdid,tol,dtflag)
    real(8),intent(in) :: time,tol
    real(8),intent(inout) :: h
    real(8),intent(out) :: hdid
    integer(c_int64_t) :: nf,nr
    integer,intent(inout) :: dtflag
    call reset_history(dtflag)
    if(step(time,h,hdid,tol,nf,nr)/=0) call fail('CUDA integration step failed')
    force_calls=force_calls+nf; rejected_steps=rejected_steps+nr
    dtflag=2
    host_current=.false.
  end subroutine
  subroutine gpu_pull(x,v)
    real(8),intent(inout) :: x(3,*),v(3,*)
    if(.not.gpu_enabled.or.host_current) return
    if(download(x,v,0)/=0) call fail('CUDA download failed')
    host_current=.true.
  end subroutine
  subroutine gpu_export(h,physical,x,v)
    real(8),intent(in) :: h
    integer,intent(in) :: physical
    real(8),intent(out) :: x(3,*),v(3,*)
    if(export_state(h,physical,x,v)/=0) call fail('CUDA coordinate export failed')
  end subroutine
  subroutine gpu_bcoord(time,jcen,n,nbig,h,m,x,v,xh,vh,ngf,ngflag,opt,bcoord)
    integer :: n,nbig,ngflag,opt(8)
    real(8) :: time,jcen(3),h,m(n),x(3,n),v(3,n),xh(3,n),vh(3,n),ngf(4,n)
    external bcoord
    if(gpu_enabled.and..not.gpu_dirty) then
      call gpu_export(h,1,xh,vh)
    else
      call bcoord(time,jcen,n,nbig,h,m,x,v,xh,vh,ngf,ngflag,opt)
    endif
  end subroutine
  subroutine gpu_old(x,v)
    real(8),intent(out) :: x(3,*),v(3,*)
    if(download(x,v,1)/=0) call fail('CUDA previous state download failed')
  end subroutine
  subroutine gpu_find_events(time,h,rcen,cap,nclo,iclo,jclo,dclo,tclo,ixv,jxv, &
      nhit,ihit,jhit,chit,dhit,thit,thit1,nowflag)
    real(8),intent(in) :: time,h,rcen
    integer,intent(in) :: cap
    integer,intent(out) :: nclo,nhit,nowflag,iclo(cap),jclo(cap),ihit(cap),jhit(cap),chit(cap)
    real(8),intent(out) :: dclo(cap),tclo(cap),ixv(6,cap),jxv(6,cap),dhit(cap),thit(cap),thit1
    type(c_ptr) :: ptr
    integer :: k
    if(find_events(time,h,rcen,ptr,event_size)/=0) call fail('CUDA encounter screening failed')
    nullify(events)
    if(event_size>0) call c_f_pointer(ptr,events,[event_size])
    nclo=0; nhit=0; nowflag=0; thit1=sign(9.9d29,h)
    do k=1,event_size
      if(iand(events(k)%kind,1)/=0) then
        nclo=nclo+1
        if(nclo>cap) call fail('Encounter buffer capacity exceeded')
        iclo(nclo)=events(k)%i; jclo(nclo)=events(k)%j
        dclo(nclo)=events(k)%distance; tclo(nclo)=events(k)%time
        ixv(:,nclo)=events(k)%xi; jxv(:,nclo)=events(k)%xj
      endif
      if(iand(events(k)%kind,2)/=0) then
        nhit=nhit+1
        if(nhit>cap) call fail('Collision buffer capacity exceeded')
        ihit(nhit)=events(k)%i; jhit(nhit)=events(k)%j
        chit(nhit)=0
        if(iand(events(k)%kind,4)/=0) chit(nhit)=1
        dhit(nhit)=events(k)%distance; thit(nhit)=events(k)%time
        nowflag=max(nowflag,events(k)%now)
        if((thit(nhit)-time)*h<(thit1-time)*h) thit1=thit(nhit)
      endif
    enddo
  end subroutine
  subroutine gpu_central(cap,nhit,jhit,thit,dhit)
    integer,intent(in) :: cap
    integer,intent(out) :: nhit,jhit(cap)
    real(8),intent(out) :: thit(cap),dhit(cap)
    integer :: k
    nhit=0
    do k=1,event_size
      if(iand(events(k)%kind,8)==0) cycle
      nhit=nhit+1
      if(nhit>cap) call fail('Central collision buffer capacity exceeded')
      jhit(nhit)=events(k)%j; thit(nhit)=events(k)%time; dhit(nhit)=events(k)%distance
    enddo
  end subroutine
end module
