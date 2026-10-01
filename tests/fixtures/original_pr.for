      subroutine mfo_pr (nbod,nbig,m,x,v,a,ngf)

c
      implicit none
      include 'mercury.inc'
c
c Input/Output
      integer nbod, nbig
      real*8 m(nbod), x(3,nbod), v(3,nbod), a(3,nbod), ngf(4,nbod)
c
c Local
      integer j
      real*8 Vt(3,nbod),D(nbod),Qpr,JH(nbod),c,beta,Vr(nbod),sw
c------------------------------------------------------------------------------
c
c     Qpr is co-efficient of PR-Drag force, in here its value is 1.

c     c = speed of light (AU/Day)

      c = 173.1

c      open (81, file='asdf6.txt',status='replace')

      do j = 2, nbod

c     To increae speed of calculation, it is required to remove the process that
c     calculate PR-effect on massive bodies(e.g. planets) because realistically
c     they do not affect by PR-effect.

      if(m(j).ne.0) then

         a(1,j)=0.d0
         a(2,j)=0.d0
         a(3,j)=0.d0

      else

c     D(j) is distance from Sun
      
      D(j) = sqrt(x(1,j)**2.d0+x(2,j)**2.d0+x(3,j)**2.d0)

c     JH = - G*M_sun*beta / (D^2 * c)
c     K2 = Gaussian Gravitational Constant Square = G*M_sun

      JH(j)= K2*ngf(4,j) / (c*D(j)**2.d0)

c     Vr is radial factor of V(velocity)

      Vr(j) = (v(1,j)*x(1,j)+v(2,j)*x(2,j)+v(3,j)*x(3,j)) / D(j)

c     sw = solar wind 
      Vt(1,j)=v(1,j)*(1.d0-x(1,j)/D(j))
      Vt(2,j)=v(2,j)*(1.d0-x(2,j)/D(j))
      Vt(3,j)=v(3,j)*(1.d0-x(3,j)/D(j))
      sw=0.3d0
c     Revised by Hangbin Jo
c     Based on Burns et al. (1979), Liou & Zook (1995), Klacka et al. (2012)
      a(1,j)= JH(j)*((c-(1.d0+sw)*2.d0*Vr(j))*x(1,j)/D(j)
     %         -(1.d0+sw)*Vt(1,j))
      a(2,j)= JH(j)*((c-(1.d0+sw)*2.d0*Vr(j))*x(2,j)/D(j)
     %         -(1.d0+sw)*Vt(2,j))
      a(3,j)= JH(j)*((c-(1.d0+sw)*2.d0*Vr(j))*x(3,j)/D(j)
     %         -(1.d0+sw)*Vt(3,j))

      endif

c       write(81,*) j, nbod, a(1,j)      

      end do

c
c------------------------------------------------------------------------------
c
      return
      end